"""
Проекция GT-семантики (18 классов, не бинарно) на картинку камеры --
тот же геометрический движок, что в occ_on_camera.py, но красим не по
зоне высоты предсказания модели, а по РЕАЛЬНОМУ классу из labels.npz
(модель бинарная, класс не предсказывает -- класс берём из метаинформации GT).
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from pyquaternion import Quaternion

from occ_baseline import VOX, X_MIN, Y_MIN, Z_MIN

sys.path.insert(0, os.path.expanduser('~/work'))
from occ_on_camera import ego_to_camera

FREE = 17
NAMES = ['others','barrier','bicycle','bus','car','constr','motorcycle','pedestrian',
         'cone','trailer','truck','road','other_flat','sidewalk','terrain',
         'manmade','vegetation']
COLORS = ['#707070','#e0559b','#33aaff','#ff8800','#0044ff','#ffaa00','#ff00aa','#ff0000',
          '#ffcc00','#a855f7','#0088ff','#c8c8c8','#88ff88','#ff9ecf','#9dd79d',
          '#e33333','#22aa22']

TOKEN = '609d5177362340458a3bfd4949cd1e64'
ROOT = 'occ3d-nus'
CAM = 'CAM_FRONT'
MIN_DEPTH, MAX_DEPTH = 1.0, 45.0

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
nusc = NuScenes('v1.0-mini', dataroot=os.path.abspath(ROOT), verbose=False)

f = glob.glob(os.path.join(ROOT, 'gts', '*', TOKEN, 'labels.npz'))[0]
sem = np.load(f)['semantics']                      # (200,200,16)

ix, iy, iz = np.nonzero(sem != FREE)
cls = sem[ix, iy, iz].astype(int)
xs = X_MIN + (ix + 0.5) * VOX
ys = Y_MIN + (iy + 0.5) * VOX
zs = Z_MIN + (iz + 0.5) * VOX
centers = np.stack([xs, ys, zs], axis=1)

colors = np.array([matplotlib.colors.to_rgb(COLORS[c]) for c in cls])

sample = nusc.get('sample', TOKEN)
lidar_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
cam_rec = nusc.get('sample_data', sample['data'][CAM])

pts_cam, cs_cam = ego_to_camera(centers, lidar_rec, cam_rec, nusc)
depths = pts_cam[2, :]
K = np.array(cs_cam['camera_intrinsic'])
points_2d = view_points(pts_cam[:3, :], K, normalize=True)

img = Image.open(os.path.join(ROOT, cam_rec['filename']))
mask = (depths > MIN_DEPTH) & (depths < MAX_DEPTH)
mask &= (points_2d[0] > 1) & (points_2d[0] < img.size[0] - 1)
mask &= (points_2d[1] > 1) & (points_2d[1] < img.size[1] - 1)
points_2d, colors_v, cls_v = points_2d[:, mask], colors[mask], cls[mask]

fig = plt.figure(figsize=(12, 7))
plt.imshow(img)
plt.scatter(points_2d[0], points_2d[1], c=colors_v, s=1.5, alpha=0.5, linewidths=0)
plt.axis('off')

# легенда только по классам, реально присутствующим в кадре
present = sorted(set(cls_v.tolist()), key=lambda c: -np.sum(cls_v == c))
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS[c], markersize=7,
                       label=f'{NAMES[c]} ({100*np.sum(cls_v==c)/len(cls_v):.0f}%)') for c in present]
plt.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=9, frameon=False)
plt.title(f'{CAM}: GT-классы occupancy (метаинфа из labels.npz, {len(present)} классов в кадре)')
plt.savefig('occ_classes_on_camera.png', bbox_inches='tight', dpi=150)
print('готово: occ_classes_on_camera.png, точек:', points_2d.shape[1], 'классов:', len(present))
