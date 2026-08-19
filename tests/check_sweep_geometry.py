import numpy as np

from occ_fusion import (
    collect_lidar_sweeps,
    X_MIN,
    Y_MIN,
    Z_MIN,
    VOXEL_SIZE,
    NX,
    NY,
    NZ,
)
from nuscenes.nuscenes import NuScenes


ROOT = "/home/voyrus/work/data/occ3d-nus"


nusc = NuScenes(
    version="v1.0-mini",
    dataroot=ROOT,
    verbose=False
)

sample = nusc.sample[0]

points = collect_lidar_sweeps(
    nusc,
    sample,
    ROOT
)

print()
print("=== SWEEP GEOMETRY ===")

print("points:", points.shape)

print("X:")
print("  min:", points[:, 0].min())
print("  max:", points[:, 0].max())
print("  mean:", points[:, 0].mean())

print("Y:")
print("  min:", points[:, 1].min())
print("  max:", points[:, 1].max())
print("  mean:", points[:, 1].mean())

print("Z:")
print("  min:", points[:, 2].min())
print("  max:", points[:, 2].max())
print("  mean:", points[:, 2].mean())


ix = np.floor(
    (points[:, 0] - X_MIN) / VOXEL_SIZE
).astype(np.int32)

iy = np.floor(
    (points[:, 1] - Y_MIN) / VOXEL_SIZE
).astype(np.int32)

iz = np.floor(
    (points[:, 2] - Z_MIN) / VOXEL_SIZE
).astype(np.int32)


valid = (
    (ix >= 0) & (ix < NX) &
    (iy >= 0) & (iy < NY) &
    (iz >= 0) & (iz < NZ)
)

print()
print("=== VOXEL COVERAGE ===")

print("inside grid:", valid.sum())
print("outside grid:", (~valid).sum())
print("inside ratio:", valid.mean())

voxels = np.stack(
    [ix[valid], iy[valid], iz[valid]],
    axis=1
)

unique_voxels = np.unique(
    voxels,
    axis=0
)

print("unique hit voxels:", len(unique_voxels))