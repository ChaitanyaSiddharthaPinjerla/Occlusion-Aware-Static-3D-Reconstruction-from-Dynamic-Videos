import numpy as np
import cv2
from pathlib import Path

folder = Path("demo_tmp/lady-running-test")

# Load depth and RGB image
depth = np.load(folder / "frame_0000.npy")
image = cv2.imread(str(folder / "frame_0000.png"))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Load camera intrinsics
K = np.loadtxt(folder / "pred_intrinsics.txt").reshape(-1, 3, 3)[0]

fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]

h, w = depth.shape

# Pixel coordinates
u, v = np.meshgrid(np.arange(w), np.arange(h))

# Back-project pixels into 3D
z = depth
x = (u - cx) * z / fx
y = (v - cy) * z / fy

# Remove invalid points
mask = np.isfinite(z) & (z > 0)

points = np.column_stack((x[mask], y[mask], z[mask]))
colors = image[mask].astype(np.uint8)

# Create colored PLY file
ply = np.empty(len(points), dtype=[
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("red", "u1"), ("green", "u1"), ("blue", "u1")
])

ply["x"], ply["y"], ply["z"] = points.T
ply["red"], ply["green"], ply["blue"] = colors.T

output = folder / "frame_0000_pointcloud.ply"

with open(output, "wb") as f:
    header = f"""ply
format binary_little_endian 1.0
element vertex {len(points)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    f.write(header.encode("ascii"))
    ply.tofile(f)

print("3D point cloud generated!")
print("Points:", len(points))
print("Saved:", output)
