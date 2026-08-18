import os
import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

"""
что сейчас сделано:

- загрузила lidar point cloud из nuScenes
- загрузила point-wise lidar segmentation
- привела исходные nuScenes классы к 17 классам TPVFormer
- перевела точки из lidar coordinates в occupancy grid 200x200x16
- сделала ray casting от lidar до каждой точки
- voxels до точки помечаем как free
- voxel с точкой помечаем ее semantic class
- сохраняем итоговые semantics и mask_lidar в npz

сейчас это базовая версия только для одного keyframe.
накопленные lidar sweeps пока не использую.
с текущим вариантом получено:
- grid: 200x200x16
- observed voxels: 73590
- gt observed voxels: 54576
- intersection с gt: 9506
- mask iou: ~8%

следующий шаг: добавить накопление lidar sweeps и перевод всех sweep points
в систему координат текущего keyframe перед ray casting.
"""

# =========================
# Occupancy grid parameters
# =========================

VOXEL_SIZE = 0.4
X_MIN, Y_MIN, Z_MIN = -40.0, -40.0, -1.0
NX, NY, NZ = 200, 200, 16

LABEL_MAPPING = {
    0: 0,
    1: 0,
    2: 7,
    3: 7,
    4: 7,
    5: 0,
    6: 7,
    7: 0,
    8: 0,
    9: 1,
    10: 0,
    11: 0,
    12: 8,
    13: 0,
    14: 2,
    15: 3,
    16: 3,
    17: 4,
    18: 5,
    19: 0,
    20: 0,
    21: 6,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    29: 0,
    30: 16,
    31: 0,
}


def load_lidar_and_segmentation(nusc, sample_token, root):
    """
    Load one LiDAR point cloud and its point-wise semantic labels.
    """

    sample = nusc.get("sample", sample_token)

    lidar_token = sample["data"]["LIDAR_TOP"]

    sample_data = nusc.get("sample_data", lidar_token)

    lidar_path = os.path.join(
        root,
        sample_data["filename"]
    )

    seg_path = os.path.join(
        root,
        "lidarseg",
        "v1.0-mini",
        lidar_token + "_lidarseg.bin"
    )

    points = np.fromfile(
        lidar_path,
        dtype=np.float32
    ).reshape(-1, 5)

    labels = np.fromfile(
        seg_path,
        dtype=np.uint8
    )

    assert len(points) == len(labels), (
        f"Number of points ({len(points)}) "
        f"does not match number of labels ({len(labels)})"
    )

    return points[:, :3], labels

def lidar_seg_to_occ(points, labels):
    """
    Convert point-wise LiDAR semantic segmentation
    into a voxel occupancy grid with free-space ray casting.

    Returns:
        semantics: (NX, NY, NZ)
        mask_lidar: (NX, NY, NZ)
    """

    semantics = np.zeros(
        (NX, NY, NZ),
        dtype=np.uint8
    )

    mask_lidar = np.zeros(
        (NX, NY, NZ),
        dtype=np.uint8
    )

    # --------------------------------------------------
    # 1. Convert LiDAR points to voxel coordinates
    # --------------------------------------------------

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

    points = points[valid]

    ix = ix[valid]
    iy = iy[valid]
    iz = iz[valid]
    labels = labels[valid]

    labels = np.array(
        [LABEL_MAPPING[int(label)] for label in labels],
        dtype=np.uint8
    )

    # --------------------------------------------------
    # 2. LiDAR sensor is at the origin of LiDAR coords
    # --------------------------------------------------

    origin = np.array([0.0, 0.0, 0.0])

    # --------------------------------------------------
    # 3. Process each LiDAR ray
    # --------------------------------------------------

    for point, label in zip(points, labels):

        distance = np.linalg.norm(point)

        if distance == 0:
            continue

        num_steps = int(
            np.ceil(distance / (VOXEL_SIZE / 2))
        )

        ts = np.linspace(
            0.0,
            1.0,
            num_steps
        )

        ray_points = (
            origin[None, :]
            + ts[:, None] * point
        )

        ray_voxels = np.floor(
            (ray_points - np.array(
                [X_MIN, Y_MIN, Z_MIN]
            )) / VOXEL_SIZE
        ).astype(np.int32)

        # Remove duplicate voxels while
        # preserving order along the ray.
        _, indices = np.unique(
            ray_voxels,
            axis=0,
            return_index=True
        )

        ray_voxels = ray_voxels[
            np.sort(indices)
        ]

        # Keep only voxels inside the grid.
        valid_voxels = (
            (ray_voxels[:, 0] >= 0) &
            (ray_voxels[:, 0] < NX) &
            (ray_voxels[:, 1] >= 0) &
            (ray_voxels[:, 1] < NY) &
            (ray_voxels[:, 2] >= 0) &
            (ray_voxels[:, 2] < NZ)
        )

        ray_voxels = ray_voxels[valid_voxels]

        if len(ray_voxels) == 0:
            continue

        hit_voxel = np.floor(
            (point - np.array([X_MIN, Y_MIN, Z_MIN]))
            / VOXEL_SIZE
        ).astype(np.int32)
        
        # --------------------------------------------------
        # 4. Everything before the hit is FREE
        # --------------------------------------------------

        if len(ray_voxels) > 1:

            # Everything before the actual hit voxel is free.
            hit_matches = np.all(
                ray_voxels == hit_voxel,
                axis=1
            )

            hit_index = np.where(hit_matches)[0]

            if len(hit_index) == 0:
                continue

            hit_index = hit_index[0]

            free_voxels = ray_voxels[:hit_index]

            free_values = semantics[
                free_voxels[:, 0],
                free_voxels[:, 1],
                free_voxels[:, 2]
            ]

            free_mask = free_values == 0

            free_voxels = free_voxels[free_mask]

            semantics[
                free_voxels[:, 0],
                free_voxels[:, 1],
                free_voxels[:, 2]
            ] = 17

            mask_lidar[
                free_voxels[:, 0],
                free_voxels[:, 1],
                free_voxels[:, 2]
            ] = 1
            
        # --------------------------------------------------
        # 5. Last voxel contains the LiDAR hit
        # --------------------------------------------------

        hit = hit_voxel

        semantics[
            hit[0],
            hit[1],
            hit[2]
        ] = label

        mask_lidar[
            hit[0],
            hit[1],
            hit[2]
        ] = 1

    return semantics, mask_lidar

def transform_lidar_to_current_ego(
    points,
    sample_data,
    current_ego_pose,
    nusc
):
    """
    Transform points from the sweep LiDAR coordinate system
    into the current keyframe ego coordinate system.
    """

    calibrated_sensor = nusc.get(
        "calibrated_sensor",
        sample_data["calibrated_sensor_token"]
    )

    ego_pose = nusc.get(
        "ego_pose",
        sample_data["ego_pose_token"]
    )

    # LiDAR -> sweep ego
    sensor_rotation = Quaternion(
        calibrated_sensor["rotation"]
    )

    sensor_translation = np.array(
        calibrated_sensor["translation"],
        dtype=np.float64
    )

    points = points @ sensor_rotation.rotation_matrix.T
    points = points + sensor_translation

    # Sweep ego -> global
    ego_rotation = Quaternion(
        ego_pose["rotation"]
    )

    ego_translation = np.array(
        ego_pose["translation"],
        dtype=np.float64
    )

    points = points @ ego_rotation.rotation_matrix.T
    points = points + ego_translation

    # Global -> current ego
    current_rotation = Quaternion(
        current_ego_pose["rotation"]
    )

    current_translation = np.array(
        current_ego_pose["translation"],
        dtype=np.float64
    )

    points = points - current_translation
    points = points @ current_rotation.rotation_matrix

    return points

def collect_lidar_sweeps(nusc, sample, root):
    """
    Collect the current LiDAR keyframe and all LiDAR sweeps
    until the next keyframe.

    All points are transformed into the current ego coordinate system.
    """

    current_lidar_token = sample["data"]["LIDAR_TOP"]

    current_sd = nusc.get(
        "sample_data",
        current_lidar_token
    )

    current_ego_pose = nusc.get(
        "ego_pose",
        current_sd["ego_pose_token"]
    )

    all_points = []
    all_labels = []

    token = current_lidar_token

    while token:

        sd = nusc.get(
            "sample_data",
            token
        )

        # Stop before the next keyframe.
        if token != current_lidar_token and sd["is_key_frame"]:
            break

        lidar_path = os.path.join(
            root,
            sd["filename"]
        )

        seg_path = os.path.join(
            root,
            "lidarseg",
            "v1.0-mini",
            token + "_lidarseg.bin"
        )

        points = np.fromfile(
            lidar_path,
            dtype=np.float32
        ).reshape(-1, 5)[:, :3]

        labels = np.fromfile(
            seg_path,
            dtype=np.uint8
        )

        assert len(points) == len(labels), (
            f"Points ({len(points)}) != "
            f"labels ({len(labels)}) for {token}"
        )

        points = transform_lidar_to_current_ego(
            points,
            sd,
            current_ego_pose,
            nusc
        )

        all_points.append(points)
        all_labels.append(labels)

        print(
            "Loaded:",
            sd["filename"],
            "points:",
            len(points),
            "keyframe:",
            sd["is_key_frame"]
        )

        token = sd["next"]

    all_points = np.concatenate(
        all_points,
        axis=0
    )

    all_labels = np.concatenate(
        all_labels,
        axis=0
    )

    print()
    print("Total accumulated points:", len(all_points))
    print("Total labels:", len(all_labels))

    return all_points, all_labels

def main():

    root = "/home/voyrus/work/data/occ3d-nus"

    nusc = NuScenes(
        version="v1.0-mini",
        dataroot=root,
        verbose=False
    )

    # First sample for debugging.
    sample = nusc.sample[0]

    points, labels = load_lidar_and_segmentation(
    nusc,
    sample["token"],
    root
)
    print("Loaded points:", points.shape)
    print("Loaded labels:", labels.shape)

    semantics, mask_lidar = lidar_seg_to_occ(
        points,
        labels
    )

    print("Occupancy shape:", semantics.shape)
    print("LiDAR mask shape:", mask_lidar.shape)

    print(
        "Observed voxels:",
        mask_lidar.sum()
    )

    output_path = (
        "/home/voyrus/work/"
        "lidar_occ_sample.npz"
    )

    np.savez_compressed(
        output_path,
        semantics=semantics,
        mask_lidar=mask_lidar
    )

    print("Saved:", output_path)

if __name__ == "__main__":
    main()