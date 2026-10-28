"""极简 3D Gaussian Splatting 训练器（gsplat 1.4 + torch 2.4 cu118）
数据: d:\\建模\\recon\\gs\\data  (images/ + sparse/0 文本模型)
用法: python train_gs.py --iters 15000
"""
import os, sys, math, time, argparse, random
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy

D = r"d:\建模\recon\gs\data"
OUT = r"d:\建模\recon\gs\out"

ap = argparse.ArgumentParser()
ap.add_argument("--iters", type=int, default=15000)
ap.add_argument("--sh", type=int, default=2)
ap.add_argument("--maxgs", type=int, default=1500000)
ap.add_argument("--out", default=OUT)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)


# ---------------- COLMAP 文本读取 ----------------
def read_model(path):
    cams, imgs = {}, []
    iid = {}
    for ln in open(os.path.join(path, "cameras.txt"), encoding="utf-8"):
        if ln.startswith("#") or not ln.strip():
            continue
        f = ln.split()
        cams[int(f[0])] = (f[1], int(f[2]), int(f[3]), [float(x) for x in f[4:]])
    lines = [l.rstrip("\n") for l in open(os.path.join(path, "images.txt"), encoding="utf-8")]
    k = 0
    while k < len(lines):
        ln = lines[k].strip()
        if not ln or ln.startswith("#"):
            k += 1; continue
        f = ln.split()
        w, x, y, z = [float(v) for v in f[1:5]]
        t = np.array([float(v) for v in f[5:8]])
        R = np.array([[1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
                      [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
                      [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]])
        imgs.append((f[9], R, t, int(f[8])))
        k += 2
    pts = []
    for ln in open(os.path.join(path, "points3D.txt"), encoding="utf-8"):
        if ln.startswith("#") or not ln.strip():
            continue
        f = ln.split()
        pts.append([float(f[1]), float(f[2]), float(f[3]), int(f[4]), int(f[5]), int(f[6])])
    return cams, imgs, np.array(pts, dtype=np.float64)


cams, imgs, P = read_model(os.path.join(D, "sparse", "0"))
cname, cW, cH, cpar = cams[1]
f0, cx0, cy0 = cpar[0], cpar[1], cpar[2]
print("images", len(imgs), "points", len(P), "camera", cname, cW, cH, flush=True)

# 归一化场景（让超参落在标准 3DGS 区间）
Cw = np.stack([-R.T @ t for (_, R, t, _) in imgs])
cen = Cw.mean(0)
S = 1.0 / np.max(np.ptp(Cw, 0) + 1e-9)
print("scene_scale", S, "center", cen.round(2), flush=True)

K = np.array([[f0, 0, cx0], [0, f0, cy0], [0, 0, 1]], np.float64)
CW = []
for (nm, R, t, cid) in imgs:
    C = -R.T @ t
    CW.append((nm, R, (C - cen) * S))

xyz = (P[:, :3] - cen) * S
rgb = P[:, 3:6] / 255.0
dev = "cuda"


# ---------------- 参数 ----------------
from scipy.spatial import cKDTree
N = len(xyz)
d1, _ = cKDTree(xyz).query(xyz, k=4)
dist = d1[:, 1:].mean(1)
dist = np.clip(dist, 1e-4, None)
print("init gaussians", N, flush=True)

C0 = 0.28209479177387814
means = torch.tensor(xyz, dtype=torch.float32, device=dev).requires_grad_(True)
scales = torch.log(torch.tensor(dist, dtype=torch.float32, device=dev)).repeat(1, 3).requires_grad_(True)
quats = torch.zeros(N, 4, device=dev); quats[:, 0] = 1.0
quats.requires_grad_(True)
opac = torch.logit(torch.full((N,), 0.1, device=dev)).requires_grad_(True)
sh0 = ((torch.tensor(rgb, dtype=torch.float32, device=dev) - 0.5) / C0).unsqueeze(1).requires_grad_(True)
NSH = (a.sh + 1) ** 2 - 1
shN = torch.zeros(N, NSH, 3, device=dev).requires_grad_(True)
params = {"means": means, "scales": scales, "quats": quats, "opacities": opac, "sh0": sh0, "shN": shN}

lrs = {"means": 1.6e-4, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2,
       "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
opts = {k: torch.optim.Adam([{"params": [v], "lr": lrs[k]}], eps=1e-15) for k, v in params.items()}
sched = torch.optim.lr_scheduler.ExponentialLR(opts["means"], gamma=0.01 ** (1.0 / a.iters))

strat = DefaultStrategy(verbose=True, refine_start_iter=500, refine_stop_iter=min(a.iters - 500, 12000),
                        reset_every=3000, refine_every=100, grow_scale3d=0.01, prune_scale3d=0.1,
                        grow_grad2d=0.0002, prune_opa=0.005)
strat.check_sanity(params, opts)
state = strat.initialize_state(scene_scale=1.0)

Kt = torch.tensor(K, dtype=torch.float32, device=dev)

# 图像懒加载缓存
cache = {}
def load(nm):
    if nm not in cache:
        img = cv2.imdecode(np.fromfile(os.path.join(D, "images", nm), dtype=np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cache[nm] = torch.from_numpy(img).float().to(dev) / 255.0
        if len(cache) > 60:
            cache.pop(next(iter(cache)))
    return cache[nm]

def ssim_loss(x, y):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_x = F.avg_pool2d(x, 5, 2); mu_y = F.avg_pool2d(y, 5, 2)
    sx = F.avg_pool2d(x * x, 5, 2) - mu_x ** 2
    sy = F.avg_pool2d(y * y, 5, 2) - mu_y ** 2
    sxy = F.avg_pool2d(x * y, 5, 2) - mu_x * mu_y
    s = ((2 * mu_x * mu_y + C1) * (2 * sxy + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (sx + sy + C2))
    return 1.0 - s.mean()

near, far = 0.01, 100.0
t0 = time.time()
for step in range(a.iters):
    nm, R, C = random.choice(CW)
    Rw = torch.tensor(R, dtype=torch.float32, device=dev)
    Ct = torch.tensor(C, dtype=torch.float32, device=dev)
    c2w = torch.eye(4, device=dev)
    c2w[:3, :3] = Rw.T
    c2w[:3, 3] = Ct
    viewmats = torch.linalg.inv(c2w)[None]
    Ks = Kt[None]
    gt = load(nm).permute(2, 0, 1)[None]

    shd = min(step // 1000, a.sh)
    colors = torch.cat([params["sh0"], params["shN"]], 1)
    renders, alphas, info = rasterization(
        means=params["means"], quats=params["quats"], scales=torch.exp(params["scales"]),
        opacities=torch.sigmoid(params["opacities"]), colors=colors,
        viewmats=viewmats, Ks=Ks, width=cW, height=cH,
        sh_degree=shd, near_plane=near, far_plane=far,
        packed=False, absgrad=False, sparse_grad=False, rasterize_mode="classic")

    strat.step_pre_backward(params, opts, state, step, info)
    loss = 0.8 * F.l1_loss(renders, gt) + 0.2 * ssim_loss(renders.permute(0, 3, 1, 2), gt)
    loss.backward()
    strat.step_post_backward(params, opts, state, step, info, packed=False)
    for o in opts.values():
        o.step(); o.zero_grad(set_to_none=True)
    sched.step()

    if step % 200 == 0:
        el = time.time() - t0
        print(f"step {step:5d} loss {loss.item():.4f} gs {len(params['means']):7d} "
              f"fps {el and (step+1)/el:.1f} eta {el/(step+1)*(a.iters-step-1)/60:.1f}min", flush=True)
    if len(params["means"]) > a.maxgs:
        print("cap reached", len(params["means"]), flush=True)
        break
    if step and step % 3000 == 0:
        torch.cuda.empty_cache()

# ---------------- 导出 PLY ----------------
with torch.no_grad():
    m = params["means"].cpu().numpy() / S + cen
    q = params["quats"].cpu().numpy()
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    sc = params["scales"].cpu().numpy() - np.log(S)  # 标准 3DGS PLY 存 log 尺度（原始单位）
    op = params["opacities"].cpu().numpy()           # 标准 3DGS PLY 存 logit 不透明度
    c0 = params["sh0"].cpu().numpy()
    cN = params["shN"].detach().cpu().numpy()

n = len(m)
props = (["x", "y", "z", "nx", "ny", "nz"] +
         [f"f_dc_{i}" for i in range(3)] +
         [f"f_rest_{i}" for i in range(cN.shape[1] * 3)] +
         ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"])
fn = os.path.join(a.out, "splat.ply")
with open(fn, "wb") as f:
    f.write(("ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % n).encode())
    for p in props:
        f.write(f"property float {p}\n".encode())
    f.write(b"end_header\n")
    arr = np.zeros((n, len(props)), np.float32)
    arr[:, 0:3] = m
    arr[:, 6:9] = c0
    fr = cN.reshape(n, -1)          # (N, K*3)
    arr[:, 9:9 + fr.shape[1]] = fr
    b = 9 + fr.shape[1]
    arr[:, b] = op
    arr[:, b + 1:b + 4] = sc
    arr[:, b + 4:b + 8] = q
    f.write(arr.tobytes())
print("saved", fn, os.path.getsize(fn) // 1024, "KB", n, "gaussians", flush=True)
torch.save({"params": {k: v.detach().cpu() for k, v in params.items()},
            "S": float(S), "cen": cen, "sh": a.sh},
           os.path.join(a.out, "params.pt"))
print("DONE", flush=True)
