"""3D 高斯泼溅训练（纯 PyTorch 光栅器，无需 CUDA 扩展）
用法: python train_torch_gs.py --iters 6000 --res 320
"""
import os, sys, time, argparse, random
import numpy as np
import cv2
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gsrender import render, quat_to_R

D = r"d:\建模\recon\gs\data"
OUT = r"d:\建模\recon\gs\out"

ap = argparse.ArgumentParser()
ap.add_argument("--iters", type=int, default=8000)
ap.add_argument("--res", type=int, default=320)
ap.add_argument("--sh", type=int, default=2)
ap.add_argument("--maxgs", type=int, default=400000)
ap.add_argument("--grow", type=float, default=0.06, help="每轮新增比例")
ap.add_argument("--init_ply", default=r"d:\建模\recon\out\cloud_fused2.ply")
ap.add_argument("--target", type=int, default=220000, help="稠密初始化点数上限")
ap.add_argument("--out", default=OUT)
ap.add_argument("--resume", default="")
ap.add_argument("--lr_means", type=float, default=3e-5)
ap.add_argument("--lr_scale", type=float, default=0.0)
ap.add_argument("--scale_max", type=float, default=0.006, help="高斯尺度上限（归一化单位）")
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
print("images", len(imgs), "points", len(P), flush=True)

Cw = np.stack([-R.T @ t for (_, R, t, _) in imgs])
cen = Cw.mean(0)
S = 1.0 / np.max(np.ptp(Cw, 0) + 1e-9)
print("scene_scale", round(float(S), 4), flush=True)

R = a.res
fx = cpar[0] * R / cW
cx = cpar[1] * R / cW
cy = cpar[2] * R / cH
K = torch.tensor([[fx, 0, cx], [0, fx, cy], [0, 0, 1]], dtype=torch.float32, device=dev)

CW = []
for (nm, Rm, t, _) in imgs:
    C = -Rm.T @ t
    c2w = np.eye(4)
    c2w[:3, :3] = Rm.T
    c2w[:3, 3] = (C - cen) * S
    vm = np.linalg.inv(c2w)
    CW.append((nm, torch.tensor(vm, dtype=torch.float32, device=dev),
               torch.tensor((C - cen) * S, dtype=torch.float32, device=dev)))

xyz = (P[:, :3] - cen) * S
rgb = P[:, 3:6] / 255.0


def load_ply_dense(path):
    dt = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                   ('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4'),
                   ('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
    with open(path, 'rb') as f:
        n = 0
        hdr = b''
        while True:
            ln = f.readline()
            hdr += ln
            s = ln.decode('ascii', 'ignore').strip()
            if s.startswith('element vertex'):
                n = int(s.split()[-1])
            if s == 'end_header':
                break
        buf = np.frombuffer(f.read(n * dt.itemsize), dtype=dt, count=n)
    p = np.stack([buf['x'], buf['y'], buf['z']], 1).astype(np.float64)
    c = np.stack([buf['r'].astype(np.float64), buf['g'].astype(np.float64),
                  buf['b'].astype(np.float64)], 1) / 255.0
    return p, c


if a.init_ply and os.path.exists(a.init_ply):
    pd, cd = load_ply_dense(a.init_ply)
    print("dense ply", len(pd), flush=True)
    pd = (pd - cen) * S
    best = None
    for vox in (0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.005, 0.007, 0.010):
        key = np.floor(pd / vox).astype(np.int64)
        _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        if len(cnt) <= a.target:
            best = (vox, inv, cnt)
            print("voxel", vox, "->", len(cnt), "points", flush=True)
            break
    if best is None:
        vox, inv, cnt = best = (0.040, inv, cnt)
    vox, inv, cnt = best
    cntf = cnt[:, None].astype(np.float64)
    psum = np.zeros((cnt.shape[0], 3))
    csum = np.zeros((cnt.shape[0], 3))
    np.add.at(psum, inv, pd)
    np.add.at(csum, inv, cd)
    xyz2 = psum / cntf
    rgb2 = np.clip(csum / cntf, 0, 1)
    # 去掉极端离群点
    d = np.linalg.norm(xyz2 - xyz2.mean(0), axis=1)
    keep = d < np.percentile(d, 99.5)
    xyz, rgb = xyz2[keep], rgb2[keep]
    print("init from dense:", len(xyz), flush=True)
    if a.target > len(xyz):
        a.target = len(xyz)


N = len(xyz)
from scipy.spatial import cKDTree
d1, _ = cKDTree(xyz).query(xyz, k=4)
dist = np.clip(d1[:, 1:].mean(1), 1e-4, None)
print("init gaussians", N, "median dist", round(float(np.median(dist)), 5), flush=True)

C0 = 0.28209479177387814
NSH = (a.sh + 1) ** 2 - 1


def mk(v, grad=True):
    t = torch.tensor(v, dtype=torch.float32, device=dev)
    return t.detach().requires_grad_(grad)


params = {
    "means": mk(xyz),
    "log_s": mk(np.log(np.repeat(dist[:, None], 3, 1))),
    "quats": mk(np.array([[1.0, 0, 0, 0]] * N)),
    "logit_o": mk(np.full(N, -2.1972)),          # sigmoid -> 0.1
    "sh0": mk(((rgb - 0.5) / C0)[:, None, :]),
    "shN": mk(np.zeros((N, NSH, 3))),
}
LRS = {"means": a.lr_means, "log_s": a.lr_scale, "quats": 1e-3, "logit_o": 5e-2,
       "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
EPSADAM = 1e-15
mom = {k: torch.zeros_like(v) for k, v in params.items()}
vel = {k: torch.zeros_like(v) for k, v in params.items()}

if a.resume:
    ck = torch.load(a.resume, map_location=dev)
    params = {k: v.detach().requires_grad_(True).to(dev) for k, v in ck["params"].items()}
    mom = {k: torch.zeros_like(v) for k, v in params.items()}
    vel = {k: torch.zeros_like(v) for k, v in params.items()}
    print("resumed", a.resume, len(params["means"]), flush=True)

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


def adam(step_t):
    for k, p in params.items():
        g = p.grad
        if g is None:
            continue
        lr = LRS[k] * (0.01 ** (step_t / a.iters) if k == "means" else 1.0)
        mom[k].mul_(0.9).add_(g, alpha=0.1)
        vel[k].mul_(0.999).addcmul_(g, g, value=0.001)
        mh = mom[k] / (1 - 0.9 ** (step_t + 1))
        vh = vel[k] / (1 - 0.999 ** (step_t + 1))
        p.data.addcdiv_(mh, vh.sqrt().add_(EPSADAM), value=-lr)


def resize_params(keep_mask):
    for k in list(params):
        params[k] = params[k].detach()[keep_mask].requires_grad_(True)
        mom[k] = mom[k][keep_mask]
        vel[k] = vel[k][keep_mask]


def grow(step, grad_acc, cnt):
    gscore = (grad_acc / cnt.clamp(min=1))
    gscore[cnt == 0] = 0
    ng = len(params["means"])
    budget = min(int(ng * a.grow), a.maxgs - ng)
    if budget <= 0:
        return 0
    top = torch.topk(gscore, min(budget, ng)).indices
    with torch.no_grad():
        m = params["means"][top]
        ls = params["log_s"][top]
        q = params["quats"][top]
        o = params["logit_o"][top]
        s0 = params["sh0"][top]
        sN = params["shN"][top]
        big = (torch.exp(ls).mean(1) > 1e-3 / S).float()[:, None]
        jitter = torch.exp(ls).mean(1, keepdim=True) * 0.6
        m2 = m + torch.randn_like(m) * jitter
        ls2 = ls - 0.693 * big
        for k, v in (("means", m2), ("log_s", ls2), ("quats", q), ("logit_o", o), ("sh0", s0), ("shN", sN)):
            params[k] = torch.cat([params[k].detach(), v], 0).requires_grad_(True)
            mom[k] = torch.cat([mom[k], torch.zeros_like(v)], 0)
            vel[k] = torch.cat([vel[k], torch.zeros_like(v)], 0)
    return len(top)


grad_acc = torch.zeros(N, device=dev)
vis_cnt = torch.zeros(N, device=dev)
near, far_ = 0.05, 200.0
t0 = time.time()
pairs_max = 6000000

for step in range(a.iters):
    nm, vm, camc = random.choice(CW)
    gt = load(nm)
    shdeg = min(step // 1000, a.sh)

    img, al, info = render(params["means"], params["quats"], params["log_s"], params["logit_o"],
                           params["sh0"], params["shN"][:, :NSH], vm, K, R, R, shdeg,
                           cam_center=camc, max_alpha_pairs=pairs_max)
    info["means2d"].retain_grad()
    loss = 0.8 * F.l1_loss(img, gt) + 0.2 * ssim_loss(img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])
    loss.backward()

    with torch.no_grad():
        g2 = info["means2d"].grad
        sel = info.get("sel")
        if g2 is not None and sel is not None and sel.numel() > 0:
            nrm = g2.norm(dim=1)
            grad_acc.index_add_(0, sel, nrm[sel])
            vis_cnt.index_add_(0, sel, 1)

    adam(step)
    if a.scale_max > 0:
        params["log_s"].data.clamp_(max=float(np.log(a.scale_max)))
    for p in params.values():
        p.grad = None

    if step >= 500 and step % 200 == 0:
        ng = grow(step, grad_acc, vis_cnt)
        grad_acc = torch.zeros(len(params["means"]), device=dev)
        vis_cnt = torch.zeros(len(params["means"]), device=dev)
        print(f"  grow +{ng} -> {len(params['means'])}", flush=True)

    if step >= 800 and step % 1000 == 0:
        with torch.no_grad():
            keep = torch.sigmoid(params["logit_o"]) > 0.01
        if int(keep.sum()) < len(keep):
            resize_params(keep)
            grad_acc = torch.zeros(len(params["means"]), device=dev)
            vis_cnt = torch.zeros(len(params["means"]), device=dev)
            print(f"  prune -> {len(params['means'])}", flush=True)

    if step % 50 == 0:
        el = time.time() - t0
        print(f"step {step:5d} loss {loss.item():.4f} gs {len(params['means']):7d} "
              f"pairs {info.get('n',0)} {el/(step+1):.2f}s/it eta {(a.iters-step-1)*el/(step+1)/60:.1f}min", flush=True)

    if step > 0 and step % 2000 == 0:
        torch.save({"params": {k: v.detach().cpu() for k, v in params.items()},
                    "S": float(S), "cen": cen, "sh": a.sh},
                   os.path.join(a.out, "latest.pt"))

torch.cuda.empty_cache()

# ---------------- 导出 ----------------
with torch.no_grad():
    m = params["means"].cpu().numpy() / S + cen
    q = params["quats"].cpu().numpy()
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
    ls = params["log_s"].cpu().numpy() - np.log(S)
    op = params["logit_o"].cpu().numpy()
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
