import os
import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

"""
что сейчас сделано:

- загрузка lidar point cloud из nuScenes
- загрузка point-wise lidar segmentation
- mapping исходных nuScenes классов в 17 классов TPVFormer
- voxelization в occupancy grid 200x200x16 с размером voxel 0.4 м
- генерация free space через ray casting от lidar origin до точки измерения
- occupied voxels получают semantic class точки
- сохранение occupancy representation в формате npz:
    - semantics
    - mask_lidar

добавлено накопление lidar sweeps:
- собираются несколько последовательных lidar кадров
- sweep points переводятся в систему координат текущего keyframe
- для каждой точки сохраняется соответствующий lidar origin,
  чтобы корректно строить ray casting из позиции сенсора

текущий результат:
- occupancy grid: 200x200x16
- keyframe points: 34688
- accumulated sweep points: 347232
- accumulated origins: 347232

следующие шаги:
- использовать сохраненные sweep origins в ray casting вместо общего origin
- проверить качество полученного occupancy относительно GT Occ3D
- подготовить формат данных для дальнейшего подключения к BEVFusion / occupancy head
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

def lidar_seg_to_occ(
    points,
    labels,
    extra_points=None,
    extra_origins=None
):
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

    if extra_points is not None:
        print(
            "Extra LiDAR points:",
            extra_points.shape
        )
        print(
            "Extra LiDAR origins:",
            extra_origins.shape
        )

        assert len(extra_points) == len(extra_origins)

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

        # --------------------------------------------------
    # Process sweep points without semantic labels
    # --------------------------------------------------

    if extra_points is not None:

        print("Processing extra LiDAR rays...")

        for point, origin in zip(extra_points, extra_origins):

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
                + ts[:, None] * (point - origin)
            )

            ray_voxels = np.floor(
                (
                    ray_points -
                    np.array([X_MIN, Y_MIN, Z_MIN])
                )
                / VOXEL_SIZE
            ).astype(np.int32)

            _, indices = np.unique(
                ray_voxels,
                axis=0,
                return_index=True
            )

            ray_voxels = ray_voxels[
                np.sort(indices)
            ]

            valid = (
                (ray_voxels[:, 0] >= 0) &
                (ray_voxels[:, 0] < NX) &
                (ray_voxels[:, 1] >= 0) &
                (ray_voxels[:, 1] < NY) &
                (ray_voxels[:, 2] >= 0) &
                (ray_voxels[:, 2] < NZ)
            )

            ray_voxels = ray_voxels[valid]

            if len(ray_voxels) == 0:
                continue

            free_voxels = ray_voxels[:-1]

            semantics[
                free_voxels[:,0],
                free_voxels[:,1],
                free_voxels[:,2]
            ] = 17

            # mask_lidar[
            #     free_voxels[:,0],
            #     free_voxels[:,1],
            #     free_voxels[:,2]
            # ] = 1

            hit = ray_voxels[-1]

            # mask_lidar[
            #     hit[0],
            #     hit[1],
            #     hit[2]
            # ] = 1

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
    all_origins = []
    # all_labels = []

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

        # seg_path = os.path.join(
        #     root,
        #     "lidarseg",
        #     "v1.0-mini",
        #     token + "_lidarseg.bin"
        # )

        points = np.fromfile(
            lidar_path,
            dtype=np.float32
        ).reshape(-1, 5)[:, :3]

        # labels = np.fromfile(
        #     seg_path,
        #     dtype=np.uint8
        # )

        # assert len(points) == len(labels), (
        #     f"Points ({len(points)}) != "
        #     f"labels ({len(labels)}) for {token}"
        # )

        points = transform_lidar_to_current_ego(
            points,
            sd,
            current_ego_pose,
            nusc
        )

        # position of this sweep LiDAR sensor
        # in the current ego coordinate system
        calibrated_sensor = nusc.get(
            "calibrated_sensor",
            sd["calibrated_sensor_token"]
        )

        ego_pose = nusc.get(
            "ego_pose",
            sd["ego_pose_token"]
        )

        sensor_translation = np.array(
            calibrated_sensor["translation"],
            dtype=np.float64
        )

        ego_rotation = Quaternion(
            ego_pose["rotation"]
        )

        ego_translation = np.array(
            ego_pose["translation"],
            dtype=np.float64
        )

        current_rotation = Quaternion(
            current_ego_pose["rotation"]
        )

        current_translation = np.array(
            current_ego_pose["translation"],
            dtype=np.float64
        )

        # LiDAR sensor -> sweep ego -> global
        origin_global = (
            sensor_translation
            @ ego_rotation.rotation_matrix.T
        ) + ego_translation

        # global -> current ego
        origin_current = (
            origin_global - current_translation
        ) @ current_rotation.rotation_matrix

        origins = np.repeat(
            origin_current[None, :],
            len(points),
            axis=0
        )

        all_points.append(points)
        all_origins.append(origins)
        # all_labels.append(labels)

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

    all_origins = np.concatenate(
        all_origins,
        axis=0
    )

    # all_labels = np.concatenate(
    #     all_labels,
    #     axis=0
    # )

    print()
    print("Total accumulated points:", len(all_points))
    print("Total accumulated origins:", len(all_origins))
    # print("Total labels:", len(all_labels))

    return all_points, all_origins

def main():

    root = "/home/voyrus/work/data/occ3d-nus"

    nusc = NuScenes(
        version="v1.0-mini",
        dataroot=root,
        verbose=False
    )

    # First sample for debugging.
    sample = nusc.sample[0]

    print()
    print("=== TEST SWEEP ACCUMULATION ===")

    sweep_points, sweep_origins = collect_lidar_sweeps(
        nusc,
        sample,
        root
    )

    print("Accumulated points shape:", sweep_points.shape)

    print("Accumulated origins shape:", sweep_origins.shape)
    print("First point:", sweep_points[0])
    print("First origin:", sweep_origins[0])

    points, labels = load_lidar_and_segmentation(
        nusc,
        sample["token"],
        root
    )

    print("Loaded points:", points.shape)
    print("Loaded labels:", labels.shape)

    semantics, mask_lidar = lidar_seg_to_occ(
        points,
        labels,
        extra_points=sweep_points,
        extra_origins=sweep_origins
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