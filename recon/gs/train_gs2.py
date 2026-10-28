"""标准 3DGS 训练（纯 PyTorch 光栅器）
- 稠密点云初始化，尺度/位置/不透明度/球谐全部可训练
- 克隆/分裂密度化、低不透明度剪枝、周期性不透明度重置
用法: python train_gs2.py --iters 8000 --res 320
"""
import os, sys, time, argparse, random
import numpy as np
import cv2
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gsrender import render

D = r"d:\建模\recon\gs\data"
C0 = 0.28209479177387814

ap = argparse.ArgumentParser()
ap.add_argument("--iters", type=int, default=8000)
ap.add_argument("--res", type=int, default=384)
ap.add_argument("--sh", type=int, default=2)
ap.add_argument("--maxgs", type=int, default=300000)
ap.add_argument("--vox", type=float, default=0.003)
ap.add_argument("--init_ply", default=r"d:\建模\recon\out\cloud_fused2.ply")
ap.add_argument("--init_ply2", default=r"d:\建模\recon\out\splat_init.ply",
                help="额外的初始化点云（Poisson 网格表面采样），用于补齐墙面/地面")
ap.add_argument("--out", default=r"d:\建模\recon\gs\out3")
ap.add_argument("--dens_thresh", type=float, default=5e-5)
ap.add_argument("--pairs", type=int, default=14000000)
ap.add_argument("--crop", type=int, default=192, help="训练时随机裁的像素窗口边长")
ap.add_argument("--maxfp", type=int, default=24)
ap.add_argument("--smax", type=float, default=0.012, help="尺度上限（归一化单位）")
ap.add_argument("--resume", default="", help="从 params_torch.pt 继续训练（用于提升分辨率微调）")
ap.add_argument("--lr_means_ft", type=float, default=1.6e-5)
ap.add_argument("--lr_scale_ft", type=float, default=5e-4)
ap.add_argument("--cam", type=int, default=-1, help="只在指定相机上训练（过拟合自检）")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = "cuda"


def read_model(path):
    cams, imgs = {}, []
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
    return cams, imgs


def load_ply_dense(path):
    dt = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4'),
                   ('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
    with open(path, 'rb') as f:
        n = 0
        while True:
            ln = f.readline()
            s = ln.decode('ascii', 'ignore').strip()
            if s.startswith('element vertex'):
                n = int(s.split()[-1])
            if s == 'end_header':
                break
        buf = np.frombuffer(f.read(n * dt.itemsize), dtype=dt, count=n)
    p = np.stack([buf['x'], buf['y'], buf['z']], 1).astype(np.float64)
    c = np.stack([buf['r'], buf['g'], buf['b']], 1).astype(np.float64) / 255.0
    return p, c


cams, imgs = read_model(os.path.join(D, "sparse", "0"))
cname, cW, cH, cpar = cams[1]
Cw = np.stack([-R.T @ t for (_, R, t) in imgs])
cen = Cw.mean(0)
S = 1.0 / np.max(np.ptp(Cw, 0) + 1e-9)
R = a.res
fx = cpar[0] * R / cW
K = torch.tensor([[fx, 0, cpar[1] * R / cW], [0, fx, cpar[2] * R / cH], [0, 0, 1]],
                 dtype=torch.float32, device=dev)

CAMS = []
for (nm, Rm, t) in imgs:
    C = -Rm.T @ t
    c2w = np.eye(4)
    c2w[:3, :3] = Rm.T
    c2w[:3, 3] = (C - cen) * S
    vm = np.linalg.inv(c2w)
    CAMS.append((nm, torch.tensor(vm, dtype=torch.float32, device=dev),
                 torch.tensor((C - cen) * S, dtype=torch.float32, device=dev)))
P, Col = load_ply_dense(a.init_ply)
if a.init_ply2 and os.path.exists(a.init_ply2):
    P2, Col2 = load_ply_dense(a.init_ply2)
    P = np.concatenate([P, P2], 0)
    Col = np.concatenate([Col, Col2], 0)
    print("merged init points", len(P), flush=True)
xyz = (P - cen) * S
key = np.floor(xyz / a.vox).astype(np.int64)
_, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
psum = np.zeros((len(cnt), 3))
csum = np.zeros((len(cnt), 3))
np.add.at(psum, inv.ravel(), xyz)
np.add.at(csum, inv.ravel(), Col)
xyz = psum / cnt[:, None]
col = np.clip(csum / cnt[:, None], 0, 1)
d = np.linalg.norm(xyz - np.median(xyz, 0), axis=1)
keep = d < np.percentile(d, 99)
xyz, col = xyz[keep], col[keep]
N = len(xyz)
from scipy.spatial import cKDTree
d1, _ = cKDTree(xyz).query(xyz, k=4)
dist = np.clip(d1[:, 1:].mean(1), 1e-4, None)
print("init gaussians", N, "vox", a.vox, "median spacing", round(float(np.median(dist)), 5), flush=True)

NSH = (a.sh + 1) ** 2 - 1


def mk(v):
    return torch.tensor(np.asarray(v), dtype=torch.float32, device=dev).requires_grad_(True)


params = {
    "means": mk(xyz),
    "log_s": mk(np.log(np.repeat((dist * 0.8)[:, None], 3, 1))),
    "quats": mk(np.tile(np.array([[1.0, 0, 0, 0]]), (N, 1))),
    "logit_o": mk(np.full(N, -2.1972)),
    "sh0": mk(((col - 0.5) / C0)[:, None, :]),
    "shN": mk(np.zeros((N, NSH, 3))),
}
LRS = {"means": 1.6e-4, "log_s": 5e-3, "quats": 1e-3, "logit_o": 5e-2,
       "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
if a.resume:
    ck = torch.load(a.resume, map_location="cpu")
    params = {k: ck["params"][k].float().to(dev).requires_grad_(True) for k in params}
    LRS["means"] = a.lr_means_ft
    LRS["log_s"] = a.lr_scale_ft
    print("resumed", a.resume, len(params["means"]), "gaussians", flush=True)
N = len(params["means"])
mom = {k: torch.zeros_like(v) for k, v in params.items()}
vel = {k: torch.zeros_like(v) for k, v in params.items()}
grad_acc = torch.zeros(N, device=dev)
vis_cnt = torch.zeros(N, device=dev)


def adam(step_t):
    for k, p in params.items():
        g = p.grad
        if g is None:
            continue
        lr = LRS[k]
        if k == "means":
            lr *= 0.01 ** (step_t / a.iters)
        mom[k].mul_(0.9).add_(g, alpha=0.1)
        vel[k].mul_(0.999).addcmul_(g, g, value=0.001)
        mh = mom[k] / (1 - 0.9 ** (step_t + 1))
        vh = vel[k] / (1 - 0.999 ** (step_t + 1))
        p.data.addcdiv_(mh, vh.sqrt().add_(1e-15), value=-lr)


def rebuild(keep_mask):
    global grad_acc, vis_cnt
    for k in list(params):
        params[k] = params[k].detach()[keep_mask].requires_grad_(True)
        mom[k] = mom[k][keep_mask]
        vel[k] = vel[k][keep_mask]
    grad_acc = grad_acc[keep_mask]
    vis_cnt = vis_cnt[keep_mask]


def densify(sel_idx):
    """克隆 + 分裂（按屏幕梯度取前 topk，保证持续生长）"""
    global grad_acc, vis_cnt
    n = len(params["means"])
    gscore = grad_acc / vis_cnt.clamp(min=1)
    gscore[vis_cnt == 0] = 0
    room = a.maxgs - n
    if room <= 0:
        return 0
    k = int(min(0.06 * n, room))
    if k <= 0:
        return 0
    top = torch.topk(gscore, k).indices
    sc = torch.exp(params["log_s"]).max(1).values
    big = sc > 0.02
    clone_idx = top[~big[top]]
    split_idx = top[big[top]]
    added = 0
    with torch.no_grad():
        if len(split_idx):
            s = split_idx
            m = params["means"][s]
            ls = params["log_s"][s] - np.log(1.6)
            q = params["quats"][s]
            o = params["logit_o"][s]
            s0 = params["sh0"][s]
            sN = params["shN"][s]
            j = torch.exp(ls).mean(1, keepdim=True) * 1.2
            m2 = m + torch.randn_like(m) * j
            for k, v in (("means", m2), ("log_s", ls), ("quats", q), ("logit_o", o), ("sh0", s0), ("shN", sN)):
                params[k] = torch.cat([params[k].detach(), v], 0).requires_grad_(True)
                mom[k] = torch.cat([mom[k], torch.zeros_like(v)], 0)
                vel[k] = torch.cat([vel[k], torch.zeros_like(v)], 0)
            added += len(s)
        if len(clone_idx):
            s = clone_idx
            j = torch.exp(params["log_s"][s]).mean(1, keepdim=True) * 0.5
            m2 = params["means"][s] + torch.randn_like(params["means"][s]) * j
            for k, v in (("means", m2), ("log_s", params["log_s"][s]), ("quats", params["quats"][s]),
                         ("logit_o", params["logit_o"][s]), ("sh0", params["sh0"][s]), ("shN", params["shN"][s])):
                params[k] = torch.cat([params[k].detach(), v], 0).requires_grad_(True)
                mom[k] = torch.cat([mom[k], torch.zeros_like(v)], 0)
                vel[k] = torch.cat([vel[k], torch.zeros_like(v)], 0)
            added += len(s)
    grad_acc = torch.zeros(len(params["means"]), device=dev)
    vis_cnt = torch.zeros(len(params["means"]), device=dev)
    return added


from gsrender import project
with torch.no_grad():
    good = []
    for c in CAMS:
        pr = project(params["means"], params["quats"], params["log_s"], c[1], K, R, R)
        if int(pr["valid"].sum()) >= 5000:
            good.append(c)
print("usable cameras", len(good), "/", len(CAMS), flush=True)
CAMS = good
if a.cam >= 0:                       # 只训练单张视图（用于过拟合自检）
    CAMS = [c for c in CAMS if c[0] == imgs[a.cam][0]]
    print("single-cam mode:", CAMS[0][0] if CAMS else "NONE", flush=True)

cache = {}


def load(nm):
    if nm not in cache:
        img = cv2.imdecode(np.fromfile(os.path.join(D, "images", nm), dtype=np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (R, R), interpolation=cv2.INTER_AREA)
        cache[nm] = torch.from_numpy(img).float().to(dev) / 255.0
        if len(cache) > 80:
            cache.pop(next(iter(cache)))
    return cache[nm]


def ssim_loss(x, y):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_x = F.avg_pool2d(x, 5, 2)
    mu_y = F.avg_pool2d(y, 5, 2)
    sx = F.avg_pool2d(x * x, 5, 2) - mu_x ** 2
    sy = F.avg_pool2d(y * y, 5, 2) - mu_y ** 2
    sxy = F.avg_pool2d(x * y, 5, 2) - mu_x * mu_y
    s = ((2 * mu_x * mu_y + C1) * (2 * sxy + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (sx + sy + C2))
    return 1.0 - s.mean()


t0 = time.time()
loss_avg = None
for step in range(a.iters):
    nm, vm, camc = random.choice(CAMS)
    gt = load(nm)
    x0 = random.randint(0, R - a.crop)
    y0 = random.randint(0, R - a.crop)
    gtc = gt[y0:y0 + a.crop, x0:x0 + a.crop]
    shdeg = int(min(a.sh, step // 1500))
    img, al, info = render(params["means"], params["quats"], params["log_s"], params["logit_o"],
                           params["sh0"], params["shN"], vm, K, R, R, shdeg,
                           cam_center=camc, max_alpha_pairs=a.pairs, max_fp=a.maxfp,
                           crop=(x0, y0, a.crop, a.crop))
    info["means2d"].retain_grad()
    loss = 0.8 * F.l1_loss(img, gtc) + 0.2 * ssim_loss(img.permute(2, 0, 1)[None], gtc.permute(2, 0, 1)[None])
    loss.backward()
    with torch.no_grad():
        g2 = info["means2d"].grad
        sel = info.get("sel")
        if g2 is not None and sel is not None and sel.numel() > 0:
            grad_acc.index_add_(0, sel, g2.norm(dim=1)[sel])
            vis_cnt.index_add_(0, sel, 1)
    adam(step)
    with torch.no_grad():
        params["log_s"].data.clamp_(min=np.log(2e-4), max=np.log(a.smax))
    for p in params.values():
        p.grad = None

    if step >= 500 and step % 100 == 0 and step < int(a.iters * 0.8):
        added = densify(None)
        if added:
            print(f"  densify +{added} -> {len(params['means'])}", flush=True)
    if step >= 500 and step % 500 == 0:
        with torch.no_grad():
            keep = torch.sigmoid(params["logit_o"]) > 0.005
        if int(keep.sum()) < len(keep):
            rebuild(keep)
            print(f"  prune -> {len(params['means'])}", flush=True)
    if step > 0 and step % 3000 == 0:
        with torch.no_grad():
            params["logit_o"].data.clamp_(max=-4.0)
        print("  opacity reset", flush=True)

    if step > 0 and step % 2000 == 0:
        torch.save({"params": {k: v.detach().cpu() for k, v in params.items()},
                    "S": float(S), "cen": cen, "sh": a.sh}, os.path.join(a.out, "latest.pt"))
        print("  checkpoint", flush=True)

    loss_avg = loss.item() if loss_avg is None else 0.98 * loss_avg + 0.02 * loss.item()
    if step % 50 == 0:
        el = time.time() - t0
        print(f"step {step:5d} loss {loss.item():.4f} avg {loss_avg:.4f} gs {len(params['means']):7d} "
              f"n {info.get('n',0)} pr {info.get('pairs',0)/1e6:.1f}M "
              f"{el/(step+1):.2f}s/it eta {(a.iters-step-1)*el/(step+1)/60:.1f}min", flush=True)

torch.cuda.empty_cache()

with torch.no_grad():
    m = params["means"].cpu().numpy() / S + cen
    q = params["quats"].cpu().numpy()
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
    ls = params["log_s"].cpu().numpy() - np.log(S)
    op = params["logit_o"].cpu().numpy()
c0 = params["sh0"].detach().cpu().numpy()
cN = params["shN"].detach().cpu().numpy()

n = len(m)
props = (["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)] +
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
    arr[:, 6:9] = c0.reshape(n, 3)
    fr = cN.reshape(n, -1)
    arr[:, 9:9 + fr.shape[1]] = fr
    b = 9 + fr.shape[1]
    arr[:, b] = op
    arr[:, b + 1:b + 4] = ls
    arr[:, b + 4:b + 8] = q
    f.write(arr.tobytes())
print("saved", fn, os.path.getsize(fn) // 1024, "KB", n, "gaussians", flush=True)
torch.save({"params": {k: v.detach().cpu() for k, v in params.items()},
            "S": float(S), "cen": cen, "sh": a.sh}, os.path.join(a.out, "params_torch.pt"))
print("DONE", flush=True)
