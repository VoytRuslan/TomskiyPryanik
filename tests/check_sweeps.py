from nuscenes.nuscenes import NuScenes


root = "/home/voyrus/work/data/occ3d-nus"

nusc = NuScenes(
    version="v1.0-mini",
    dataroot=root,
    verbose=False
)

sample = nusc.get(
    "sample",
    "ca9a282c9e77460f8360f564131a8af5"
)

token = sample["data"]["LIDAR_TOP"]

print("CURRENT SAMPLE")
print("sample token:", sample["token"])
print("lidar token:", token)
print()

i = 0

while token:
    sd = nusc.get("sample_data", token)

    print(
        i,
        "timestamp =", sd["timestamp"],
        "keyframe =", sd["is_key_frame"],
        "file =", sd["filename"]
    )

    token = sd["next"]
    i += 1

    # Останавливаемся на следующем keyframe
    if token:
        next_sd = nusc.get("sample_data", token)

        if next_sd["is_key_frame"]:
            print()
            print("NEXT KEYFRAME FOUND")
            print("token:", token)
            print("timestamp:", next_sd["timestamp"])
            print("file:", next_sd["filename"])
            break