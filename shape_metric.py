"""
Метрика формы (идея ментора): усреднённый шаблон объекта в объектной системе
координат (выровнен по yaw, нормирован, ресемплен в 8x8x16), схожесть
предсказания с шаблоном -- отдельно от IoU/Dice, которые меряют позицию,
а не форму.

Извлечение инстансов -- два разных способа, т.к. в Occ3D нет instance-id:
    - barrier, traffic_cone, pedestrian -- у них есть настоящие 3D-боксы
      с yaw в аннотациях nuScenes (global -> ego через ego_pose)
    - manmade, vegetation -- фоновые классы, объектов nuScenes для них нет:
      connected components (scipy.ndimage.label) + PCA по XY для оценки yaw

Шаблоны строятся ТОЛЬКО на train-сценах (сплит по сценам, не по кадрам --
см. находку про утечку в CLAUDE.md; для этого скрипта сплит по сценам
реализован здесь же, отдельно от тренировочных скриптов). Схожесть pred-
инстанса с шаблоном -- "мягкий" Dice: 2*sum(template*pred)/(sum(template)+
sum(pred)), шаблон непрерывный [0,1], порог не нужен.

Не применимо к road/sidewalk/terrain -- это не дискретные объекты, а сплошная
поверхность на всю сцену, "форма дороги" не имеет смысла.

Сейчас поддержан только бинарный occ_baseline.py: модель не знает семантических
подклассов, поэтому pred-маска внутри бокса/blob'а -- "что-то предсказано
занятым", а не конкретно предсказанный класс.

Запуск:
    cd ~/work/data
    python ~/work/shape_metric.py --ckpt occ_baseline.pth
"""
import argparse, glob, math, os, sys, time
import numpy as np
import torch
from scipy import ndimage
from pyquaternion import Quaternion

sys.path.insert(0, os.path.expanduser('~/work'))
from occ_baseline import lidar_to_bev, OccNet, VOX, X_MIN, Y_MIN, Z_MIN, NX, NY, NZ

OUT_SHAPE = (8, 8, 16)

BOX_CATEGORIES = {
    1: ['movable_object.barrier'],
    8: ['movable_object.trafficcone'],
    7: ['human.pedestrian.adult', 'human.pedestrian.child', 'human.pedestrian.construction_worker',
        'human.pedestrian.police_officer', 'human.pedestrian.personal_mobility',
        'human.pedestrian.stroller', 'human.pedestrian.wheelchair'],
}
BOX_CATEGORY_TO_CLASS = {name: cid for cid, names in BOX_CATEGORIES.items() for name in names}
BLOB_CLASSES = [15, 16]
CLASS_NAMES = {1: 'barrier', 7: 'pedestrian', 8: 'traffic_cone', 15: 'manmade', 16: 'vegetation'}


def idx_range(lo_m, hi_m, origin, n):
    lo_i = max(0, int(math.floor((lo_m - origin) / VOX)))
    hi_i = min(n, int(math.ceil((hi_m - origin) / VOX)))
    return lo_i, hi_i


def normalize_resample(local_xyz, out_shape=OUT_SHAPE):
    """(N,3) точки в выровненной по yaw, центрированной системе -> нормирует
    под unit box (масштаб-инвариантно) и ресемплит в бинарную сетку out_shape."""
    lo = local_xyz.min(axis=0)
    hi = local_xyz.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    norm = (local_xyz - lo) / span
    idx = np.clip(np.floor(norm * np.array(out_shape)).astype(np.int64), 0, np.array(out_shape) - 1)
    grid = np.zeros(out_shape, dtype=np.float32)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
    return grid


def soft_dice(template, instance_bool):
    """Dice шаблона (непрерывный [0,1]) и бинарного инстанса -- без порога."""
    inter = float((template * instance_bool).sum())
    return 2 * inter / max(float(template.sum() + instance_bool.sum()), 1e-6)


# ============================================================
#  ИЗВЛЕЧЕНИЕ ИНСТАНСОВ: barrier / traffic_cone / pedestrian (реальные боксы)
# ============================================================
def global_box_to_ego(nusc, ann_token, sample_data_token):
    box = nusc.get_box(ann_token)                       # box в глобальной системе
    sd = nusc.get('sample_data', sample_data_token)
    ego_pose = nusc.get('ego_pose', sd['ego_pose_token'])
    box.translate(-np.array(ego_pose['translation']))
    box.rotate(Quaternion(ego_pose['rotation']).inverse)
    return box                                           # теперь в ego


def extract_box_instance(sem_grid, pred_grid, box, class_id, min_voxels):
    """sem_grid/pred_grid: (NX,NY,NZ). pred_grid=None -> pred_shape не считаем
    (для train-сцен, где предсказание не нужно). Возвращает (gt_shape, pred_shape)
    или None, если внутри бокса меньше min_voxels вокселей нужного класса."""
    R = box.orientation.rotation_matrix
    yaw = math.atan2(R[1, 0], R[0, 0])
    cx, cy, cz = box.center
    w, l, h = box.wlh
    pad = 0.3
    r = max(l, w) / 2 + pad

    ix_lo, ix_hi = idx_range(cx - r, cx + r, X_MIN, NX)
    iy_lo, iy_hi = idx_range(cy - r, cy + r, Y_MIN, NY)
    iz_lo, iz_hi = idx_range(cz - h / 2 - pad, cz + h / 2 + pad, Z_MIN, NZ)
    if ix_lo >= ix_hi or iy_lo >= iy_hi or iz_lo >= iz_hi:
        return None

    xs = X_MIN + (np.arange(ix_lo, ix_hi) + 0.5) * VOX
    ys = Y_MIN + (np.arange(iy_lo, iy_hi) + 0.5) * VOX
    zs = Z_MIN + (np.arange(iz_lo, iz_hi) + 0.5) * VOX
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')

    dx, dy, dz = gx - cx, gy - cy, gz - cz
    cos_y, sin_y = math.cos(-yaw), math.sin(-yaw)
    lx = dx * cos_y - dy * sin_y                          # локальная система бокса:
    ly = dx * sin_y + dy * cos_y                           # x вдоль yaw, y поперёк
    lz = dz

    inside = (np.abs(lx) < l / 2 + pad) & (np.abs(ly) < w / 2 + pad) & (np.abs(lz) < h / 2 + pad)

    sem_block = sem_grid[ix_lo:ix_hi, iy_lo:iy_hi, iz_lo:iz_hi]
    gt_mask = inside & (sem_block == class_id)
    if gt_mask.sum() < min_voxels:
        return None
    gt_shape = normalize_resample(np.stack([lx[gt_mask], ly[gt_mask], lz[gt_mask]], axis=1))

    pred_shape = None
    if pred_grid is not None:
        pred_block = pred_grid[ix_lo:ix_hi, iy_lo:iy_hi, iz_lo:iz_hi]
        pred_mask = inside & pred_block
        if pred_mask.sum() >= 1:
            pred_shape = normalize_resample(np.stack([lx[pred_mask], ly[pred_mask], lz[pred_mask]], axis=1))
        else:
            pred_shape = np.zeros(OUT_SHAPE, dtype=np.float32)

    return gt_shape, pred_shape


# ============================================================
#  ИЗВЛЕЧЕНИЕ ИНСТАНСОВ: manmade / vegetation (connected components + PCA)
# ============================================================
def extract_blob_instances(sem_grid, pred_grid, class_id, min_voxels):
    """Возвращает список (gt_shape, pred_shape) по каждой связной компоненте
    класса class_id крупнее min_voxels вокселей."""
    mask = sem_grid == class_id
    if not mask.any():
        return []
    labels, n = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=np.int32))

    xs = X_MIN + (np.arange(NX) + 0.5) * VOX
    ys = Y_MIN + (np.arange(NY) + 0.5) * VOX
    zs = Z_MIN + (np.arange(NZ) + 0.5) * VOX
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')

    out = []
    for lbl in range(1, n + 1):
        blob = labels == lbl
        if blob.sum() < min_voxels:
            continue

        bx, by, bz = gx[blob], gy[blob], gz[blob]
        cx, cy, cz0 = bx.mean(), by.mean(), bz.mean()

        # PCA по XY -- главная горизонтальная осьblob'а как оценка yaw
        # (нет аннотированного бокса, нет другого способа узнать ориентацию)
        xy = np.stack([bx - cx, by - cy], axis=1)
        cov = xy.T @ xy / max(len(xy) - 1, 1)
        eigval, eigvec = np.linalg.eigh(cov)
        main = eigvec[:, int(np.argmax(eigval))]
        yaw = math.atan2(main[1], main[0])
        cos_y, sin_y = math.cos(-yaw), math.sin(-yaw)

        dx, dy = bx - cx, by - cy
        lx = dx * cos_y - dy * sin_y
        ly = dx * sin_y + dy * cos_y
        lz = bz - cz0
        gt_shape = normalize_resample(np.stack([lx, ly, lz], axis=1))

        pred_shape = None
        if pred_grid is not None:
            pred_mask = blob & pred_grid
            if pred_mask.sum() >= 1:
                pdx, pdy = gx[pred_mask] - cx, gy[pred_mask] - cy
                plx = pdx * cos_y - pdy * sin_y
                ply = pdx * sin_y + pdy * cos_y
                plz = gz[pred_mask] - cz0
                pred_shape = normalize_resample(np.stack([plx, ply, plz], axis=1))
            else:
                pred_shape = np.zeros(OUT_SHAPE, dtype=np.float32)

        out.append((gt_shape, pred_shape))

    return out


# ============================================================
#  ИНФЕРЕНС
# ============================================================
def load_lidar_points(nusc, root, token):
    sample = nusc.get('sample', token)
    sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    path = os.path.join(root, sd['filename'])
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :3]
    cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
    return pts @ Quaternion(cs['rotation']).rotation_matrix.T + np.array(cs['translation'])


def infer_occupancy(net, nusc, root, token, dev):
    pts = load_lidar_points(nusc, root, token)
    bev = lidar_to_bev(pts)
    with torch.no_grad():
        logits = net(torch.from_numpy(bev).unsqueeze(0).to(dev))
    return logits.argmax(1)[0].permute(1, 2, 0).cpu().numpy().astype(bool)   # (X,Y,Z)


# ============================================================
#  ОСНОВНОЙ ЦИКЛ
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='occ3d-nus')
    ap.add_argument('--ckpt', default='occ_baseline.pth')
    ap.add_argument('--limit', type=int, default=None, help='кадров всего (по умолчанию -- весь датасет)')
    ap.add_argument('--val-frac', type=float, default=0.2, help='доля СЦЕН на val')
    ap.add_argument('--min-voxels', type=int, default=6, help='мин. вокселей, чтобы считать инстанс валидным')
    args = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', dev)

    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes('v1.0-mini', dataroot=os.path.abspath(args.root), verbose=False)

    files = sorted(glob.glob(os.path.join(args.root, 'gts', '*', '*', 'labels.npz')))
    if args.limit:
        files = files[:args.limit]

    scenes = sorted({f.split(os.sep)[-3] for f in files})
    rng = np.random.RandomState(0)
    order = rng.permutation(len(scenes))
    n_val = max(1, int(len(scenes) * args.val_frac))
    val_scenes = {scenes[i] for i in order[:n_val]}
    print(f'кадров: {len(files)}, сцен: {len(scenes)}, val-сцен: {len(val_scenes)} ({sorted(val_scenes)})')

    net = OccNet(num_classes=2).to(dev)
    net.load_state_dict(torch.load(args.ckpt, map_location=dev))
    net.eval()

    template_sum = {c: np.zeros(OUT_SHAPE, dtype=np.float64) for c in CLASS_NAMES}
    template_cnt = {c: 0 for c in CLASS_NAMES}
    val_shapes = {c: [] for c in CLASS_NAMES}

    t0 = time.time()
    n_val_frames = 0
    for f in files:
        token = os.path.basename(os.path.dirname(f))
        scene = f.split(os.sep)[-3]
        is_val = scene in val_scenes
        sem = np.load(f)['semantics']
        sample = nusc.get('sample', token)

        pred_grid = None
        if is_val:
            try:
                pred_grid = infer_occupancy(net, nusc, args.root, token, dev)
                n_val_frames += 1
            except Exception:
                pred_grid = None                          # нет лидара для кадра -- пропускаем pred

        for ann_token in sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)
            cls_id = BOX_CATEGORY_TO_CLASS.get(ann['category_name'])
            if cls_id is None:
                continue
            box = global_box_to_ego(nusc, ann_token, sample['data']['LIDAR_TOP'])
            res = extract_box_instance(sem, pred_grid if is_val else None, box, cls_id, args.min_voxels)
            if res is None:
                continue
            gt_shape, pred_shape = res
            if is_val:
                if pred_shape is not None:
                    val_shapes[cls_id].append(pred_shape)
            else:
                template_sum[cls_id] += gt_shape
                template_cnt[cls_id] += 1

        for cls_id in BLOB_CLASSES:
            for gt_shape, pred_shape in extract_blob_instances(sem, pred_grid if is_val else None,
                                                                cls_id, args.min_voxels):
                if is_val:
                    if pred_shape is not None:
                        val_shapes[cls_id].append(pred_shape)
                else:
                    template_sum[cls_id] += gt_shape
                    template_cnt[cls_id] += 1

    print(f'обработка: {time.time()-t0:.0f}s (val-кадров с инференсом: {n_val_frames})\n')

    print(f'{"класс":14s} {"train-инст.":>12s} {"val-инст.":>10s} {"similarity":>11s}')
    for c in CLASS_NAMES:
        if template_cnt[c] < 3:
            print(f'{CLASS_NAMES[c]:14s} {template_cnt[c]:12d}  -- мало train-инстансов, шаблон ненадёжен')
            continue
        template = (template_sum[c] / template_cnt[c]).astype(np.float32)
        scores = [soft_dice(template, s) for s in val_shapes[c]]
        mean_score = float(np.mean(scores)) if scores else float('nan')
        print(f'{CLASS_NAMES[c]:14s} {template_cnt[c]:12d} {len(val_shapes[c]):10d} {mean_score:11.4f}')


if __name__ == '__main__':
    main()
