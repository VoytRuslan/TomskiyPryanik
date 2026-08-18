import os
import numpy as np

from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion


ROOT = "/home/voyrus/work/data/occ3d-nus"


def load_points(root, filename):
    path = os.path.join(root, filename)

    points = np.fromfile(
        path,
        dtype=np.float32
    ).reshape(-1, 5)

    return points[:, :3]


def lidar_to_ego(points, calibrated_sensor):
    """
    LiDAR coordinates -> ego coordinates.
    """

    rotation = Quaternion(
        calibrated_sensor["rotation"]
    )

    translation = np.array(
        calibrated_sensor["translation"]
    )

    points = points @ rotation.rotation_matrix.T
    points = points + translation

    return points


def main():

    nusc = NuScenes(
        version="v1.0-mini",
        dataroot=ROOT,
        verbose=False
    )

    # Current keyframe.
    sample = nusc.get(
        "sample",
        "ca9a282c9e77460f8360f564131a8af5"
    )

    lidar_token = sample["data"]["LIDAR_TOP"]

    # Current LiDAR sample_data.
    sd = nusc.get(
        "sample_data",
        lidar_token
    )

    # Calibration of current LiDAR.
    cs = nusc.get(
        "calibrated_sensor",
        sd["calibrated_sensor_token"]
    )

    points = load_points(
        ROOT,
        sd["filename"]
    )

    ego_points = lidar_to_ego(
        points,
        cs
    )

    print("Original LiDAR points:")
    print(points.shape)

    print()

    print("First point in LiDAR coordinates:")
    print(points[0])

    print()

    print("Same point in ego coordinates:")
    print(ego_points[0])

    print()

    print("LiDAR -> ego translation:")
    print(cs["translation"])

    print()

    print("LiDAR -> ego rotation:")
    print(cs["rotation"])


if __name__ == "__main__":
    main()