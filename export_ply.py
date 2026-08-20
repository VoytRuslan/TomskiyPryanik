"""
СЕРВЕР (без дисплея). Прогоняет модель на одном кадре и сохраняет
воксели как .ply — GT и предсказание отдельно, с раскраской по зонам высоты.

Никакого GUI здесь не вызывается: только запись файлов.

Запуск:
    cd ~/work/data
    python ~/work/export_ply.py --ckpt occ_baseline.pth --idx 0
"""
import argparse, os, sys
import numpy as np
import open3d as o3d
import torch

sys.path.insert(0, os.path.expanduser('~/work'))
from occ_baseline import Occ3DDataset, OccNet, VOX, Z_MIN, NZ, z_index

# цвета зон: под колёсами / габарит / над габаритом
COLORS = {'low': [0.35, 0.75, 0.35],    # зелёный — можно пропустить под колёсами
          'mid': [0.90, 0.25, 0.25],    # красный — реально мешает
          'high': [0.25, 0.45, 0.95]}   # синий — можно проехать под


def zone_colors(zi_arr, i_clear, i_top):
    c = np.zeros((len(zi_arr), 3))
    c[zi_arr < i_clear] = COLORS['low']
    c[(zi_arr >= i_clear) & (zi_arr < i_top)] = COLORS['mid']
    c[zi_arr >= i_top] = COLORS['high']
    return c


def save_voxels(mask, path, i_clear, i_top):
    """mask: (X, Y, Z) bool -> цветной .ply"""
    ijk = np.argwhere(mask)
    if len(ijk) == 0:
        print('пусто, нечего писать:', path)
        return
    # в метры; z считаем от Z_MIN
    xyz = ijk.astype(np.float64) * VOX
    xyz[:, 2] += Z_MIN

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz)
    pc.colors = o3d.utility.Vector3dVector(zone_colors(ijk[:, 2], i_clear, i_top))
    o3d.io.write_point_cloud(path, pc)
    print(f'{path}: {len(ijk)} вокселей')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='occ3d-nus')
    ap.add_argument('--ckpt', default='occ_baseline.pth')
    ap.add_argument('--idx', type=int, default=0, help='индекс кадра в датасете')
    ap.add_argument('--out', default='.')
    args = ap.parse_args()

    i_clear, i_top = z_index(0.2), z_index(2.0)

    nusc = None
    try:
        from nuscenes.nuscenes import NuScenes
        nusc = NuScenes('v1.0-mini', dataroot=os.path.abspath(args.root), verbose=False)
    except Exception as e:
        print('devkit недоступен:', e)

    ds = Occ3DDataset(args.root, binary=True, nusc=nusc)
    print(f'кадров всего: {len(ds)}, берём {args.idx}')
    bev, tgt = ds[args.idx]

    net = OccNet(num_classes=2)
    net.load_state_dict(torch.load(args.ckpt, map_location='cpu'))
    net.eval()
    with torch.no_grad():
        pred = net(bev.unsqueeze(0)).argmax(1)[0]        # (Z, H, W)

    gt_mask = tgt.numpy() == 1                            # (H, W, Z); -100 (ignore) не считаем занятым
    pred_mask = pred.numpy().transpose(1, 2, 0).astype(bool)

    save_voxels(gt_mask, os.path.join(args.out, 'gt.ply'), i_clear, i_top)
    save_voxels(pred_mask, os.path.join(args.out, 'pred.ply'), i_clear, i_top)

    # где ошиблись — отдельным файлом, это самое интересное для защиты
    save_voxels(gt_mask & ~pred_mask, os.path.join(args.out, 'missed.ply'), i_clear, i_top)
    save_voxels(pred_mask & ~gt_mask, os.path.join(args.out, 'extra.ply'), i_clear, i_top)

    inter = (gt_mask & pred_mask).sum()
    union = (gt_mask | pred_mask).sum()
    print(f'IoU этого кадра: {inter / max(union, 1):.4f}')


if __name__ == '__main__':
    main()