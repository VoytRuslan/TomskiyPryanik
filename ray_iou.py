"""
RayIoU -- метрика из SparseOcc (github.com/MCG-NJU/SparseOcc, loaders/ray_metrics.py).

Кидает фиксированный веер лучей (имитация лидара: ~10 углов по pitch x 360 по
azimuth) из позиции LIDAR_TOP, для каждого луча ищет первое попадание в занятый
воксель, сравнивает pred vs gt по расстоянию (пороги 1/2/4 м, как в оригинале)
и классу, считает IoU по классам и усредняет.

Ray-voxel intersection в оригинале -- отдельное CUDA-расширение (lib/dvr/dvr.cpp
+ dvr.cu), собирается через torch cpp_extension при импорте. Сознательно НЕ
тащим его -- новая непроверенная зависимость на хрупком окружении перед защитой.
Вместо этого -- DDA-марш вдоль луча с фиксированным шагом на чистом PyTorch:
та же математика (лучи, пороги, IoU-формула из calc_rayiou), медленнее их CUDA-
версии, но это eval, не тренировка -- считается один раз по датасету.

Отличие от оригинала: single-frame origin (позиция LIDAR_TOP в ego текущего
кадра), а не до 8 origin'ов с соседних sweeps -- у нас нет задачи 4D forecasting,
только single-frame occupancy. Геометрия сетки (VOX=0.4, [-40,-40,-1,40,40,5.4])
идентична оригиналу -- это и есть наша сетка Occ3D-nuScenes.

Сейчас поддержан только бинарный occ_baseline.py (OccNet) -- класс "occupied"
против free. Для честной посемантической RayIoU (18 классов) или для fusion/
bevfusion моделей нужно расширить main() по аналогии, не делал за неимением
времени перед защитой.

Запуск:
    cd ~/work/data
    python ~/work/ray_iou.py --ckpt occ_baseline.pth --limit 30
"""
import argparse, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.expanduser('~/work'))
from occ_baseline import Occ3DDataset, OccNet, VOX, X_MIN, Y_MIN, Z_MIN, NX, NY, NZ, FREE

THRESHOLDS = [1.0, 2.0, 4.0]        # метры, как в оригинале
MAX_RANGE = 120.0                    # с запасом больше диагонали грида (~113 м)
STEP = VOX / 2                       # шаг марша -- не пропустить воксель


def generate_lidar_rays():
    """Дословный порт generate_lidar_rays() из SparseOcc -- фиксированный веер
    направлений, имитирующий угловое разрешение лидара."""
    pitch_angles = []
    for k in range(10):
        angle = math.pi / 2 - math.atan(k + 1)
        pitch_angles.append(-angle)

    while pitch_angles[-1] < 0.21:
        delta = pitch_angles[-1] - pitch_angles[-2]
        pitch_angles.append(pitch_angles[-1] + delta)

    rays = []
    for pitch in pitch_angles:
        for az_deg in np.arange(0, 360, 1):
            az = np.deg2rad(az_deg)
            x = np.cos(pitch) * np.cos(az)
            y = np.cos(pitch) * np.sin(az)
            z = np.sin(pitch)
            rays.append((x, y, z))
    return np.array(rays, dtype=np.float32)


def _ray_box_exit(origin, dirs, lo, hi):
    """Расстояние выхода лучей из AABB [lo,hi] (слэб-метод). origin: (3,) внутри
    box -- у нас так и есть, LIDAR_TOP рядом с центром сетки."""
    device = dirs.device
    lo_t = torch.as_tensor(lo, dtype=torch.float32, device=device)
    hi_t = torch.as_tensor(hi, dtype=torch.float32, device=device)
    origin_t = torch.as_tensor(origin, dtype=torch.float32, device=device)

    eps = 1e-8
    par = dirs.abs() < eps                              # лучи, параллельные оси i
    d = torch.where(par, torch.full_like(dirs, eps), dirs)

    t_lo = (lo_t - origin_t) / d
    t_hi = (hi_t - origin_t) / d
    exit_i = torch.maximum(t_lo, t_hi)                   # положительный корень = выход по этой оси
    exit_i = torch.where(par, torch.full_like(dirs, MAX_RANGE), exit_i)
    return exit_i.min(dim=1).values.clamp(min=0.1, max=MAX_RANGE)


def cast_rays(class_grid, origin, dirs, free_id, device):
    """
    class_grid: (NX,NY,NZ) класс-id по вокселям (np.uint8/int, свободно=free_id)
    origin: (3,) позиция сенсора в ego-системе
    dirs: (N,3) единичные направления лучей (на device)
    Возвращает (dist, cls): дистанция и класс первого занятого вокселя на луче,
    либо (дистанция выхода из грида, free_id), если луч ни во что не попал.
    """
    grid_t = torch.from_numpy(class_grid.astype(np.int64)).to(device)
    occ = grid_t != free_id

    t_exit = _ray_box_exit(origin, dirs, [X_MIN, Y_MIN, Z_MIN],
                            [X_MIN + NX * VOX, Y_MIN + NY * VOX, Z_MIN + NZ * VOX])
    origin_t = torch.as_tensor(origin, dtype=torch.float32, device=device)

    N = dirs.shape[0]
    hit = torch.zeros(N, dtype=torch.bool, device=device)
    hit_t = t_exit.clone()

    n_steps = int(math.ceil(MAX_RANGE / STEP))
    for s in range(1, n_steps + 1):
        t = s * STEP
        active = (~hit) & (t <= t_exit)
        if not active.any():
            break
        pts = origin_t + dirs * t
        ix = ((pts[:, 0] - X_MIN) / VOX).long()
        iy = ((pts[:, 1] - Y_MIN) / VOX).long()
        iz = ((pts[:, 2] - Z_MIN) / VOX).long()
        inside = active & (ix >= 0) & (ix < NX) & (iy >= 0) & (iy < NY) & (iz >= 0) & (iz < NZ)
        if not inside.any():
            continue
        idx = inside.nonzero(as_tuple=True)[0]
        occ_here = occ[ix[idx], iy[idx], iz[idx]]
        newly = idx[occ_here]
        hit_t[newly] = t
        hit[newly] = True

    pts = origin_t + dirs * hit_t[:, None]
    ix = ((pts[:, 0] - X_MIN) / VOX).long().clamp(0, NX - 1)
    iy = ((pts[:, 1] - Y_MIN) / VOX).long().clamp(0, NY - 1)
    iz = ((pts[:, 2] - Z_MIN) / VOX).long().clamp(0, NZ - 1)
    cls = grid_t[ix, iy, iz]
    cls = torch.where(hit, cls, torch.full_like(cls, free_id))
    return hit_t, cls


def calc_rayiou(pred_grids, gt_grids, origins, class_ids, free_id=FREE,
                 thresholds=THRESHOLDS, device='cuda'):
    """Дословный порт calc_rayiou() из SparseOcc: TP/пороги/IoU по классам."""
    rays = torch.from_numpy(generate_lidar_rays()).to(device)
    n_th, n_cls = len(thresholds), len(class_ids)
    gt_cnt = np.zeros(n_cls)
    pred_cnt = np.zeros(n_cls)
    tp_cnt = np.zeros((n_th, n_cls))

    for pred_grid, gt_grid, origin in zip(pred_grids, gt_grids, origins):
        pred_dist, pred_cls = cast_rays(pred_grid, origin, rays, free_id, device)
        gt_dist, gt_cls = cast_rays(gt_grid, origin, rays, free_id, device)

        valid = gt_cls != free_id                       # оцениваем только реальные лучи GT
        pred_dist, pred_cls = pred_dist[valid], pred_cls[valid]
        gt_dist, gt_cls = gt_dist[valid], gt_cls[valid]

        l1 = (pred_dist - gt_dist).abs().cpu().numpy()
        pred_cls_np, gt_cls_np = pred_cls.cpu().numpy(), gt_cls.cpu().numpy()

        for j, th in enumerate(thresholds):
            tp_dist = l1 < th
            for i, c in enumerate(class_ids):
                m_pred = pred_cls_np == c
                m_gt = gt_cls_np == c
                if j == 0:
                    gt_cnt[i] += m_gt.sum()
                    pred_cnt[i] += m_pred.sum()
                tp_cnt[j, i] += (m_pred & m_gt & tp_dist).sum()

    iou = [tp_cnt[j] / np.maximum(gt_cnt + pred_cnt - tp_cnt[j], 1) for j in range(n_th)]
    return iou, gt_cnt, pred_cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='occ3d-nus')
    ap.add_argument('--ckpt', default='occ_baseline.pth')
    ap.add_argument('--limit', type=int, default=30,
                     help='сколько кадров оценивать -- дороже, чем IoU/Dice, весь датасет будет медленно')
    args = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', dev)

    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes('v1.0-mini', dataroot=os.path.abspath(args.root), verbose=False)

    ds = Occ3DDataset(args.root, binary=True, limit=args.limit, nusc=nusc)

    net = OccNet(num_classes=2).to(dev)
    net.load_state_dict(torch.load(args.ckpt, map_location=dev))
    net.eval()

    pred_grids, gt_grids, origins = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for i in range(len(ds)):
            f = ds.files[i]
            token = os.path.basename(os.path.dirname(f))
            bev, _ = ds[i]
            pred = net(bev.unsqueeze(0).to(dev)).argmax(1)[0]        # (Z,H,W) 0/1
            pred_grid = pred.permute(1, 2, 0).cpu().numpy()          # (X,Y,Z)
            # бинарная модель не знает подклассов -- кодируем как "класс 1"
            # (занято) / FREE, чтобы формат совпадал с семантической сеткой
            pred_sem = np.where(pred_grid > 0, 1, FREE).astype(np.uint8)

            gt_sem = np.load(f)['semantics']                          # (X,Y,Z) 0..17
            gt_bin = np.where(gt_sem != FREE, 1, FREE).astype(np.uint8)

            sample = nusc.get('sample', token)
            sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
            origin = np.array(cs['translation'], dtype=np.float32)   # LIDAR_TOP в ego

            pred_grids.append(pred_sem)
            gt_grids.append(gt_bin)
            origins.append(origin)
    print(f'инференс + подготовка: {time.time()-t0:.0f}s, кадров: {len(pred_grids)}')

    t0 = time.time()
    iou, gt_cnt, pred_cnt = calc_rayiou(pred_grids, gt_grids, origins,
                                        class_ids=[1], free_id=FREE, device=dev)
    print(f'ray casting: {time.time()-t0:.0f}s')

    vals = [iou[j][0] for j in range(len(THRESHOLDS))]
    print(f'RayIoU@1: {vals[0]:.4f}  RayIoU@2: {vals[1]:.4f}  RayIoU@4: {vals[2]:.4f}')
    print(f'RayIoU (среднее по порогам): {np.mean(vals):.4f}')
    print(f'лучей GT="occupied": {int(gt_cnt[0])}  лучей pred="occupied": {int(pred_cnt[0])}')


if __name__ == '__main__':
    main()
