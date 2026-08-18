import os
import numpy as np

from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion


ROOT = "/home/voyrus/work/data/occ3d-nus"

CURRENT_SAMPLE = "ca9a282c9e77460f8360f564131a8af5"

SWEEP_TOKEN = "0cedf1d2d652468d92d23491136b5d15"


def load_points(filename):
    path = os.path.join(ROOT, filename)

    points = np.fromfile(
        path,
        dtype=np.float32
    ).reshape(-1, 5)

    return points[:, :3]


def transform_lidar_to_ego(points, calibrated_sensor):
    rotation = Quaternion(
        calibrated_sensor["rotation"]
    )

    translation = np.array(
        calibrated_sensor["translation"]
    )

    points = points @ rotation.rotation_matrix.T
    points = points + translation

    return points


def transform_ego_to_global(points, ego_pose):
    rotation = Quaternion(
        ego_pose["rotation"]
    )

    translation = np.array(
        ego_pose["translation"]
    )

    points = points @ rotation.rotation_matrix.T
    points = points + translation

    return points


def transform_global_to_ego(points, ego_pose):
    rotation = Quaternion(
        ego_pose["rotation"]
    )

    translation = np.array(
        ego_pose["translation"]
    )

    points = points - translation
    points = points @ rotation.rotation_matrix

    return points


def main():

    nusc = NuScenes(
        version="v1.0-mini",
        dataroot=ROOT,
        verbose=False
    )

    # ---------------------------------------
    # Current keyframe
    # ---------------------------------------

    current_sample = nusc.get(
        "sample",
        CURRENT_SAMPLE
    )

    current_lidar_token = current_sample["data"]["LIDAR_TOP"]

    current_sd = nusc.get(
        "sample_data",
        current_lidar_token
    )

    current_cs = nusc.get(
        "calibrated_sensor",
        current_sd["calibrated_sensor_token"]
    )

    current_pose = nusc.get(
        "ego_pose",
        current_sd["ego_pose_token"]
    )

    # ---------------------------------------
    # Sweep
    # ---------------------------------------

    sweep_sd = nusc.get(
        "sample_data",
        SWEEP_TOKEN
    )

    sweep_cs = nusc.get(
        "calibrated_sensor",
        sweep_sd["calibrated_sensor_token"]
    )

    sweep_pose = nusc.get(
        "ego_pose",
        sweep_sd["ego_pose_token"]
    )

    # ---------------------------------------
    # Load sweep points
    # ---------------------------------------

    points = load_points(
        sweep_sd["filename"]
    )

    print("Sweep points:", points.shape)

    # ---------------------------------------
    # Sweep LiDAR -> sweep ego
    # ---------------------------------------

    points = transform_lidar_to_ego(
        points,
        sweep_cs
    )

    # ---------------------------------------
    # Sweep ego -> global
    # ---------------------------------------

    points = transform_ego_to_global(
        points,
        sweep_pose
    )

    # ---------------------------------------
    # Global -> current ego
    # ---------------------------------------

    points = transform_global_to_ego(
        points,
        current_pose
    )

    print("Transformed points:", points.shape)

    print()
    print("First transformed point:")
    print(points[0])

    print()
    print("Current ego pose:")
    print("translation:", current_pose["translation"])
    print("rotation:", current_pose["rotation"])

    print()
    print("Sweep ego pose:")
    print("translation:", sweep_pose["translation"])
    print("rotation:", sweep_pose["rotation"])


if __name__ == "__main__":
    main()