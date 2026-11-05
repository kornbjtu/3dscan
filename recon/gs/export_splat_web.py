"""把训练好的 3DGS 参数导出成浏览器可漫游的单文件 HTML（WebGL2 实时泼溅）

用法: python export_splat_web.py --pt d:\建模\recon\gs\out\params_torch.pt --out d:\建模\recon\out\splat_view.html
每个 splat 32 字节:
  [0:12]  pos   xyz  f32
  [12:24] scale xyz  f32 (线性尺度)
  [24:28] quat    u8 x4 (w,x,y,z, 映射 v/255*2-1)
  [28:31] color   u8 x3
  [31]    opacity u8
"""
import os, sys, base64, argparse
import numpy as np
import torch

TMPL = r"C:\Users\zlll-\AppData\Local\Temp\splat_tpl.html"
C0 = 0.28209479177387814

ap = argparse.ArgumentParser()
ap.add_argument("--pt", default=r"d:\建模\recon\gs\out\params_torch.pt")
ap.add_argument("--sparse", default=r"d:\建模\recon\gs\data\sparse\0")
ap.add_argument("--out", default=r"d:\建模\recon\out\splat_view.html")
ap.add_argument("--target", type=float, default=8.0, help="场景归一化后的最大跨度(单位)")
ap.add_argument("--tmpl", default=TMPL)
ap.add_argument("--maxsplats", type=int, default=0, help="限制导出的高斯数量(按不透明度和尺度筛)")
ap.add_argument("--start_cam", type=int, default=0, help="起始相机（对应 images.txt 顺序）")
a = ap.parse_args()


def read_images(path):
    imgs = []
    lines = [l.rstrip("\n") for l in open(os.path.join(path, "images.txt"), encoding="utf-8")]
    k = 0
    while k < len(lines):
        ln = lines[k].strip()
        if not ln or ln.startswith("#"):
            k += 1
            continue
        f = ln.split()
        w, x, y, z = [float(v) for v in f[1:5]]
        t = np.array([float(v) for v in f[5:8]])
        R = np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                      [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                      [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])
        imgs.append((f[9], R, t))
        k += 2
    return imgs


def q_from_R(Rm):
    """旋转矩阵 -> 四元数 (w,x,y,z)"""
    tr = Rm[0, 0] + Rm[1, 1] + Rm[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (Rm[2, 1] - Rm[1, 2]) / s
        y = (Rm[0, 2] - Rm[2, 0]) / s
        z = (Rm[1, 0] - Rm[0, 1]) / s
    elif Rm[0, 0] > Rm[1, 1] and Rm[0, 0] > Rm[2, 2]:
        s = np.sqrt(1.0 + Rm[0, 0] - Rm[1, 1] - Rm[2, 2]) * 2
        w = (Rm[2, 1] - Rm[1, 2]) / s
        x = 0.25 * s
        y = (Rm[0, 1] + Rm[1, 0]) / s
        z = (Rm[0, 2] + Rm[2, 0]) / s
    elif Rm[1, 1] > Rm[2, 2]:
        s = np.sqrt(1.0 + Rm[1, 1] - Rm[0, 0] - Rm[2, 2]) * 2
        w = (Rm[0, 2] - Rm[2, 0]) / s
        x = (Rm[0, 1] + Rm[1, 0]) / s
        y = 0.25 * s
        z = (Rm[1, 2] + Rm[2, 1]) / s
    else:
        s = np.sqrt(1.0 + Rm[2, 2] - Rm[0, 0] - Rm[1, 1]) * 2
        w = (Rm[1, 0] - Rm[0, 1]) / s
        x = (Rm[0, 2] + Rm[2, 0]) / s
        y = (Rm[1, 2] + Rm[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def R_from_q(q):
    w, x, y, z = q
    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                     [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                     [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


# ---------- 读训练结果 ----------
ck = torch.load(a.pt, map_location="cpu", weights_only=False)
p = ck["params"]
S, cen = float(ck["S"]), np.asarray(ck["cen"])

means = p["means"].numpy().astype(np.float64) / S + cen     # 世界坐标
q = p["quats"].numpy().astype(np.float64)
q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-9
sc = np.exp(p["log_s"].numpy().astype(np.float64)) / S      # 世界尺度
op = 1.0 / (1.0 + np.exp(-p["logit_o"].numpy().astype(np.float64)))
c0 = p["sh0"].numpy().reshape(len(means), 3).astype(np.float64)
col = np.clip(c0 * C0 + 0.5, 0, 1)
n_all = len(means)
print("gaussians", n_all)

# ---------- 朝向：用相机平均 up / forward 建立世界->视图 旋转 ----------
imgs = read_images(a.sparse)
Cw = np.stack([-R.T @ t for (_, R, t) in imgs])
up_mean = np.zeros(3)
fw_mean = np.zeros(3)
for (_, R, t) in imgs:
    Rt = R.T
    up_mean += -Rt[:, 1]        # 相机坐标 y 向下 -> 取反
    fw_mean += Rt[:, 2]         # 相机 z 向前
Y = up_mean / (np.linalg.norm(up_mean) + 1e-12)
f0 = fw_mean - (fw_mean @ Y) * Y
if np.linalg.norm(f0) < 1e-6:
    f0 = np.array([0.0, 0.0, -1.0]) - (np.array([0.0, 0.0, -1.0]) @ Y) * Y
f0 /= np.linalg.norm(f0)
Zv = -f0                        # 视图系里相机看向 -Z
Xv = np.cross(f0, Y)
Xv /= np.linalg.norm(Xv)
Yv = np.cross(Zv, Xv)
Rw2v = np.stack([Xv, Yv, Zv])   # 行向量
print("up_mean", np.round(Y, 3), "fwd", np.round(f0, 3))

center = Cw.mean(0)
span = float(np.max(np.ptp(Cw, 0)) + 1e-9)
k = a.target / span
print("scene span", round(span, 3), "scale", round(k, 4))

pos = (means - center) @ Rw2v.T * k
cams_v = (Cw - center) @ Rw2v.T * k
scl = sc * k

# 高斯朝向：R_v = Rw2v R R^T Rw2v
qv = np.empty_like(q)
for i in range(n_all):
    Rg = R_from_q(q[i])
    qv[i] = q_from_R(Rw2v @ Rg @ Rw2v.T)

# 起始相机：--start_cam >= 0 指定；< 0 时自动选覆盖高斯最多的相机
si = a.start_cam
if si < 0:
    sub = pos[::max(1, n_all // 20000)]
    best, si = -1, 0
    for i, (_, Rm, t) in enumerate(imgs):
        Rv = Rw2v @ Rm.T                     # view<-camera 旋转
        fw = Rv[:, 2]
        if abs(fw[1]) > 0.45:                # 跳过俯仰过大的面片（朝上/朝下的立方体面）
            continue
        pv = sub @ Rw2v.T
        z = (pv - cams_v[i]) @ fw            # 相机坐标系下的前向分量
        c = int(((z > 0) & (z < 3.0)).sum())
        if c > best:
            best, si = c, i
    print("auto start_cam", si, "in-front pts", best)
else:
    si = si % len(imgs)
start = cams_v[si]
fw0 = Rw2v @ imgs[si][1].T @ np.array([0.0, 0.0, 1.0])   # 相机 +Z 轴 -> 观察方向
fw0 /= np.linalg.norm(fw0) + 1e-12
yaw0 = float(np.arctan2(fw0[0], -fw0[2]))
pitch0 = float(np.arcsin(np.clip(fw0[1], -1, 1)))
print("start", np.round(start, 2), "yaw", round(yaw0, 3), "pitch", round(pitch0, 3))

# ---------- 可选裁剪 ----------
if a.maxsplats and a.maxsplats < n_all:
    score = op / (np.cbrt(scl.prod(1)) + 1e-9)
    keep = np.argsort(-score)[:a.maxsplats]
    pos, qv, scl, op, col = pos[keep], qv[keep], scl[keep], op[keep], col[keep]
    n = len(keep)
    print("trimmed to", n)
else:
    n = n_all

# ---------- 打包 32 字节/splat ----------
buf = np.zeros((n, 32), np.uint8)
buf[:, 0:12] = pos.astype(np.float32).view(np.uint8).reshape(n, 12)
buf[:, 12:24] = scl.astype(np.float32).view(np.uint8).reshape(n, 12)
qq = np.clip(np.round((qv + 1.0) * 0.5 * 255), 0, 255).astype(np.uint8)
buf[:, 24:28] = qq
buf[:, 28:31] = np.clip(np.round(col * 255), 0, 255).astype(np.uint8)
buf[:, 31] = np.clip(np.round(op * 255), 0, 255).astype(np.uint8)

data = base64.b64encode(buf.tobytes()).decode("ascii")
print("payload", round(len(data) / 1048576, 2), "MB (base64)")

html = open(a.tmpl, encoding="utf-8").read()
html = (html.replace("__N__", str(n))
            .replace("__DATA__", data)
            .replace("__SX__", f"{start[0]:.4f}")
            .replace("__SY__", f"{start[1]:.4f}")
            .replace("__SZ__", f"{start[2]:.4f}")
            .replace("__YAW__", f"{yaw0:.4f}")
            .replace("__PITCH__", f"{pitch0:.4f}"))
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "w", encoding="utf-8") as f:
    f.write(html)
print("saved", a.out, round(os.path.getsize(a.out) / 1048576, 2), "MB")
