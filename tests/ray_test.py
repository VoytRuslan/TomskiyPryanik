import numpy as np


VOXEL_SIZE = 0.4

X_MIN, Y_MIN, Z_MIN = -40.0, -40.0, -1.0

NX, NY, NZ = 200, 200, 16


def point_to_voxel(point):
    x, y, z = point

    ix = int(np.floor((x - X_MIN) / VOXEL_SIZE))
    iy = int(np.floor((y - Y_MIN) / VOXEL_SIZE))
    iz = int(np.floor((z - Z_MIN) / VOXEL_SIZE))

    return np.array([ix, iy, iz])


def ray_voxels(origin, point):
    """
    Return voxel indices along the ray
    from LiDAR origin to the measured point.
    """

    distance = np.linalg.norm(point - origin)

    num_steps = int(np.ceil(distance / (VOXEL_SIZE / 2)))

    ts = np.linspace(0.0, 1.0, num_steps)

    points = origin[None, :] + ts[:, None] * (
        point - origin
    )

    voxels = np.array([
        point_to_voxel(p)
        for p in points
    ])

    # Remove duplicate voxels.
    _, indices = np.unique(
        voxels,
        axis=0,
        return_index=True
    )

    voxels = voxels[np.sort(indices)]

    # Keep voxels inside occupancy grid.
    valid = (
        (voxels[:, 0] >= 0) & (voxels[:, 0] < NX) &
        (voxels[:, 1] >= 0) & (voxels[:, 1] < NY) &
        (voxels[:, 2] >= 0) & (voxels[:, 2] < NZ)
    )

    return voxels[valid]


origin = np.array([0.0, 0.0, 0.0])

point = np.array([
    10.0,
    5.0,
    1.0
])

voxels = ray_voxels(origin, point)

print("Point:", point)
print("Number of voxels:", len(voxels))
print("First voxels:")
print(voxels[:10])
print("Last voxel:")
print(voxels[-1])