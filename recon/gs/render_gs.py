"""用训练好的高斯泼溅离线渲染预览图（纯 torch 光栅器）
用法: python render_gs.py [--every 40] [--res 640]
"""
import os, sys, argparse
import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gsrender import render

D = r"d:\建模\recon\gs\data"
OUT = r"d:\建模\recon\gs\out"

ap = argparse.ArgumentParser()
ap.add_argument("--every", type=int, default=40)
ap.add_argument("--res", type=int, default=640)
ap.add_argument("--pt", default=os.path.join(OUT, "params_torch.pt"))
ap.add_argument("--tag", default="")
a = ap.parse_args()


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
    return cams, imgs


cams, imgs = read_model(os.path.join(D, "sparse", "0"))
cname, cW, cH, cpar = cams[1]
R = a.res
K = torch.tensor([[cpar[0] * R / cW, 0, cpar[1] * R / cW],
                  [0, cpar[0] * R / cH, cpar[2] * R / cH],
                  [0, 0, 1]], dtype=torch.float32, device="cuda")

ck = torch.load(a.pt, map_location="cuda")
p, S, cen, shd = ck["params"], ck["S"], np.asarray(ck["cen"]), ck["sh"]
print("gaussians", len(p["means"]), "res", R, flush=True)

os.makedirs(os.path.join(OUT, "render"), exist_ok=True)
for i in range(0, len(imgs), a.every):
    nm, Rm, t, _ = imgs[i]
    C = (-Rm.T @ t - cen) * S
    c2w = np.eye(4)
    c2w[:3, :3] = Rm.T
    c2w[:3, 3] = C
    vm = torch.tensor(np.linalg.inv(c2w), dtype=torch.float32, device="cuda")
    cc = torch.tensor(C, dtype=torch.float32, device="cuda")
    with torch.no_grad():
        img, al, _ = render(p["means"].cuda(), p["quats"].cuda(), p["log_s"].cuda(), p["logit_o"].cuda(),
                            p["sh0"].cuda(), p["shN"].cuda(), vm, K, R, R, shd,
                            cam_center=cc, max_alpha_pairs=20000000)
    rimg = (img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)[:, :, ::-1]
    gt = cv2.imdecode(np.fromfile(os.path.join(D, "images", nm), dtype=np.uint8), cv2.IMREAD_COLOR)
    if gt.shape[0] != R:
        gt = cv2.resize(gt, (R, R))
    cmpimg = np.concatenate([rimg, gt], 1)
    cv2.putText(cmpimg, "GS", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(cmpimg, "GT", (R + 8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    fn = os.path.join(OUT, "render", f"cmp{a.tag}_{i:04d}.png")
    cv2.imencode(".png", cmpimg)[1].tofile(fn)
    print("wrote", fn, flush=True)
print("DONE", flush=True)
