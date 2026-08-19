"""
Проекция предсказанных occupancy-вокселей на картинку камеры.

Геометрия трансформации -- по мотивам ~/lidar_to_camera.py (Вадим):
ego(t_лидара) -> global -> ego(t_камеры) -> камера-сенсор -> пиксели (K).
Отличие: вместо сырых точек лидара проецируем ЦЕНТРЫ занятых вокселей из
предсказания модели, шаг 1 (лидар-сенсор -> ego) не нужен -- сетка occupancy
уже в ego-системе. Раскраска по зоне высоты, как в export_ply.py:
    зелёный -- под колёсами (можно проехать)
    красный -- в габарите (реальная помеха)
    синий   -- над габаритом (можно проехать под)

Запуск:
    cd ~/work/data
    python ~/work/occ_on_camera.py --ckpt occ_bevfusion.pth --cam CAM_FRONT
"""
import argparse, os, sys
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

from occ_baseline import VOX, X_MIN, Y_MIN, Z_MIN, z_index
from occ_bevfusion import Occ3DCamDataset, BEVFusionOccNet

# та же палитра, что в export_ply.py -- единый визуальный язык проекта
COLORS = {'low': (0.35, 0.75, 0.35), 'mid': (0.90, 0.25, 0.25), 'high': (0.25, 0.45, 0.95)}


def ego_to_camera(centers_xyz, lidar_rec, cam_rec, nusc):
    """centers_xyz: (N,3) в ego-системе кадра (= система occupancy-сетки).
    Повторяет шаги 2-4 из ~/lidar_to_camera.py (Вадим); шаг 1 (лидар-сенсор
    -> ego) пропущен -- воксели уже в ego."""
    pts = centers_xyz.T.astype(np.float64)                       # (3, N)

    ego_lidar = nusc.get('ego_pose', lidar_rec['ego_pose_token'])
    pts = Quaternion(ego_lidar['rotation']).rotation_matrix @ pts
    pts = pts + np.array(ego_lidar['translation'])[:, None]      # ego(t_лидара) -> global

    ego_cam = nusc.get('ego_pose', cam_rec['ego_pose_token'])
    pts = pts - np.array(ego_cam['translation'])[:, None]
    pts = Quaternion(ego_cam['rotation']).rotation_matrix.T @ pts  # global -> ego(t_камеры)

    cs_cam = nusc.get('calibrated_sensor', cam_rec['calibrated_sensor_token'])
    pts = pts - np.array(cs_cam['translation'])[:, None]
    pts = Quaternion(cs_cam['rotation']).rotation_matrix.T @ pts   # ego(t_камеры) -> камера

    return pts, cs_cam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='occ3d-nus')
    ap.add_argument('--ckpt', default='occ_bevfusion.pth')
    ap.add_argument('--cam', default='CAM_FRONT')
    ap.add_argument('--token', default='609d5177362340458a3bfd4949cd1e64',
                     help='токен сэмпла; по умолчанию -- кейс из scene-0916, '
                          'топ по ложным блокировкам (см. viz_good.py)')
    ap.add_argument('--min-depth', type=float, default=1.0)
    ap.add_argument('--max-depth', type=float, default=45.0,
                     help='отсечь далёкие воксели -- иначе горизонт забивается точками')
    ap.add_argument('--out', default='occ_on_camera.png')
    args = ap.parse_args()

    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.geometry_utils import view_points
    nusc = NuScenes('v1.0-mini', dataroot=os.path.abspath(args.root), verbose=False)

    ds = Occ3DCamDataset(args.root, binary=True, nusc=nusc)
    idx = next(i for i, f in enumerate(ds.files) if args.token in f)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = BEVFusionOccNet(num_classes=2).to(dev)
    net.load_state_dict(torch.load(args.ckpt, map_location=dev))
    net.eval()

    bev, tgt, imgs, intrins, cam2ego = ds[idx]
    with torch.no_grad():
        logits = net(bev.unsqueeze(0).to(dev), imgs.unsqueeze(0).to(dev),
                     intrins.unsqueeze(0).to(dev), cam2ego.unsqueeze(0).to(dev))
        pred = logits.argmax(1)[0].cpu().numpy()          # (Z, NX, NY), 0/1

    iz, ix, iy = np.nonzero(pred)
    xs = X_MIN + (ix + 0.5) * VOX
    ys = Y_MIN + (iy + 0.5) * VOX
    zs = Z_MIN + (iz + 0.5) * VOX
    centers = np.stack([xs, ys, zs], axis=1)               # (N,3) ego

    i_clear, i_top = z_index(0.2), z_index(2.0)
    colors = np.empty((len(iz), 3))
    colors[iz < i_clear] = COLORS['low']
    colors[(iz >= i_clear) & (iz < i_top)] = COLORS['mid']
    colors[iz >= i_top] = COLORS['high']

    sample = nusc.get('sample', args.token)
    lidar_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    cam_rec = nusc.get('sample_data', sample['data'][args.cam])

    pts_cam, cs_cam = ego_to_camera(centers, lidar_rec, cam_rec, nusc)
    depths = pts_cam[2, :]
    K = np.array(cs_cam['camera_intrinsic'])
    points_2d = view_points(pts_cam[:3, :], K, normalize=True)

    img = Image.open(os.path.join(args.root, cam_rec['filename']))
    mask = (depths > args.min_depth) & (depths < args.max_depth)
    mask &= (points_2d[0] > 1) & (points_2d[0] < img.size[0] - 1)
    mask &= (points_2d[1] > 1) & (points_2d[1] < img.size[1] - 1)
    points_2d, colors_v = points_2d[:, mask], colors[mask]

    plt.figure(figsize=(12, 7))
    plt.imshow(img)
    plt.scatter(points_2d[0], points_2d[1], c=colors_v, s=1.5, alpha=0.45, linewidths=0)
    plt.axis('off')
    plt.title(f'{args.cam}: предсказанная occupancy-сетка ({os.path.basename(args.ckpt)})\n'
              'зелёный — под колёсами, красный — габарит, синий — над габаритом')
    plt.savefig(args.out, bbox_inches='tight', dpi=150)
    print(f'готово: {args.out}, точек в кадре: {points_2d.shape[1]}')


if __name__ == '__main__':
    main()
