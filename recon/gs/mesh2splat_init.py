"""从 Poisson 网格按面积均匀采样点，生成 3DGS 初始化点云
用法: python mesh2splat_init.py [--mesh ...] [--n 400000] [--out ...]
"""
import os, sys, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--mesh", default=r"d:\建模\recon\out\mesh_photo.ply")
ap.add_argument("--n", type=int, default=400000)
ap.add_argument("--out", default=r"d:\建模\recon\out\splat_init.ply")
ap.add_argument("--rotx", type=float, default=90.0, help="绕X轴旋转角度（网格与点云坐标系差）")
a = ap.parse_args()


def read_ply(path):
    with open(path, "rb") as f:
        props, nv, nf = [], 0, 0
        cur = None
        while True:
            ln = f.readline().decode("ascii", "ignore").strip()
            if ln.startswith("element vertex"):
                cur = "v"; nv = int(ln.split()[-1])
            elif ln.startswith("element face"):
                cur = "f"; nf = int(ln.split()[-1])
            elif ln.startswith("property") and cur == "v":
                props.append(ln.split()[-1])
            elif ln == "end_header":
                break
        dt = []
        for p in props:
            nm = {"red": "r", "green": "g", "blue": "b"}.get(p, p)
            if p in ("x", "y", "z", "nx", "ny", "nz"):
                dt.append((nm, "<f8"))
            elif nm in ("r", "g", "b"):
                dt.append((nm, "u1"))
            else:
                dt.append((nm, "<f4"))
        vb = np.frombuffer(f.read(nv * np.dtype(dt).itemsize), dtype=dt, count=nv)
        fdt = np.dtype([("k", "u1"), ("i0", "<i4"), ("i1", "<i4"), ("i2", "<i4")])
        fb = np.frombuffer(f.read(nf * fdt.itemsize), dtype=fdt, count=nf)
    v = np.stack([vb["x"], vb["y"], vb["z"]], 1)
    c = np.stack([vb["r"], vb["g"], vb["b"]], 1).astype(np.float64) / 255.0
    tri = np.stack([fb["i0"], fb["i1"], fb["i2"]], 1).astype(np.int64)
    return v, c, tri


v, c, tri = read_ply(a.mesh)
print("mesh verts", len(v), "tris", len(tri), flush=True)
p0, p1, p2 = v[tri[:, 0]], v[tri[:, 1]], v[tri[:, 2]]
area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
area = np.clip(area, 1e-12, None)
prob = area / area.sum()
idx = np.random.choice(len(tri), a.n, p=prob)
r1 = np.sqrt(np.random.rand(a.n))
r2 = np.random.rand(a.n)
trip = tri[idx]
w0 = 1 - r1
w1 = r1 * (1 - r2)
w2 = r1 * r2
pts = w0[:, None] * v[trip[:, 0]] + w1[:, None] * v[trip[:, 1]] + w2[:, None] * v[trip[:, 2]]
cols = w0[:, None] * c[trip[:, 0]] + w1[:, None] * c[trip[:, 1]] + w2[:, None] * c[trip[:, 2]]
cols = np.clip(cols, 0, 1)
if a.rotx:
    t_ = np.radians(a.rotx)
    Rx = np.array([[1, 0, 0], [0, np.cos(t_), -np.sin(t_)], [0, np.sin(t_), np.cos(t_)]])
    pts = pts @ Rx.T

# 去重：体素下采样到约 n 点，避免三角形内采样过密
vox = 0.012
key = np.floor(pts / vox).astype(np.int64)
_, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
acc_p = np.zeros((len(cnt), 3)); acc_c = np.zeros((len(cnt), 3))
np.add.at(acc_p, inv.ravel(), pts); np.add.at(acc_c, inv.ravel(), cols)
pts = acc_p / cnt[:, None]; cols = acc_c / cnt[:, None]
print("after voxel", vox, "->", len(pts), flush=True)

n = len(pts)
dt = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4'),
               ('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
arr = np.zeros(n, dt)
arr['x'], arr['y'], arr['z'] = pts[:, 0], pts[:, 1], pts[:, 2]
arr['r'] = np.round(cols[:, 0] * 255).astype(np.uint8)
arr['g'] = np.round(cols[:, 1] * 255).astype(np.uint8)
arr['b'] = np.round(cols[:, 2] * 255).astype(np.uint8)
with open(a.out, "wb") as f:
    f.write(("ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % n).encode())
    for p in ("float x", "float y", "float z", "float nx", "float ny", "float nz",
              "uchar red", "uchar green", "uchar blue"):
        f.write(f"property {p}\n".encode())
    f.write(b"end_header\n")
    f.write(arr.tobytes())
print("wrote", a.out, n, "points", flush=True)
