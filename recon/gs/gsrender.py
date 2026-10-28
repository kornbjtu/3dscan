"""纯 PyTorch 的 3D 高斯泼溅光栅器（不依赖 CUDA 扩展）
支持: 各向异性协方差、球谐颜色、按深度排序的前向 alpha 合成、自动求导。
"""
import torch
import torch.nn.functional as F

EPS = 1e-9


def quat_to_R(q):
    q = q / (q.norm(dim=1, keepdim=True) + EPS)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.stack([
        torch.stack([1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y], 1),
        torch.stack([2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x], 1),
        torch.stack([2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y], 1),
    ], 1)
    return R


def sh_basis(dirs, deg):
    """dirs: (N,3) 单位向量, 返回 (N, (deg+1)^2)"""
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    out = [torch.full_like(x, 0.28209479177387814)]
    if deg >= 1:
        out += [-0.4886025119029199 * y, 0.4886025119029199 * z, -0.4886025119029199 * x]
    if deg >= 2:
        out += [
            1.0925484305920792 * x * y,
            -1.0925484305920792 * y * z,
            0.31539156525252005 * (3 * z * z - 1),
            -1.0925484305920792 * x * z,
            0.5462742152960396 * (x * x - y * y),
        ]
    return torch.stack(out, 1)


def project(means, quats, log_scales, viewmat, K, W, H, eps2d=0.3, min_z=0.02, max_rad=None):
    """投影为 2D 高斯。返回 dict（含保留掩码）"""
    Rq = quat_to_R(quats)
    S = torch.exp(log_scales)
    M = Rq * S[:, None, :]
    Sigma = M @ M.transpose(1, 2)

    Rv, tv = viewmat[:3, :3], viewmat[:3, 3]
    p = means @ Rv.T + tv
    z = p[:, 2]
    ok = z > min_z
    zc = z.clamp(min=min_z)

    Sig_c = Rv @ Sigma @ Rv.T
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    J = torch.zeros(means.shape[0], 2, 3, device=means.device, dtype=means.dtype)
    J[:, 0, 0] = fx / zc
    J[:, 0, 2] = -fx * p[:, 0] / zc ** 2
    J[:, 1, 1] = fy / zc
    J[:, 1, 2] = -fy * p[:, 1] / zc ** 2
    cov2 = J @ Sig_c @ J.transpose(1, 2)
    cov2 = cov2 + torch.eye(2, device=means.device, dtype=means.dtype)[None] * eps2d

    det = cov2[:, 0, 0] * cov2[:, 1, 1] - cov2[:, 0, 1] * cov2[:, 1, 0]
    ok = ok & (det > 1e-12)

    mu = fx * p[:, 0] / zc + cx
    mv = fy * p[:, 1] / zc + cy
    tr = cov2[:, 0, 0] + cov2[:, 1, 1]
    disc = torch.sqrt(torch.clamp((cov2[:, 0, 0] - cov2[:, 1, 1]) ** 2 + 4 * cov2[:, 0, 1] ** 2, min=0))
    rad = 3.0 * torch.sqrt(torch.clamp(0.5 * (tr + disc), min=0))
    ok = ok & (rad > 0.35) & (mu > -rad) & (mu < W + rad) & (mv > -rad) & (mv < H + rad)
    if max_rad is not None:
        ok = ok & (rad < max_rad)

    return dict(mu=mu, mv=mv, cov2=cov2, det=det, rad=rad, depth=z, valid=ok, means2d=torch.stack([mu, mv], 1))


def render(means, quats, log_scales, logit_opac, sh0, shN, viewmat, K, W, H, shdeg,
           cam_center=None, max_fp=18, max_alpha_pairs=None, max_rad=None, crop=None):
    """返回 (img[hc,wc,3], alpha[hc,wc], info)

    crop=(x0,y0,cw,ch) 只光栅化该像素窗口（训练用），其余 splat 全部保留参与合成，
    避免"随机丢弃 splat"导致画面偏暗、模型靠放大高斯补偿的偏差。
    """
    N = means.shape[0]
    if max_rad is None:
        max_rad = 0.15 * min(W, H)
    pr = project(means, quats, log_scales, viewmat, K, W, H, max_rad=max_rad)
    ok = pr["valid"]
    if crop is None:
        x0, y0, cw, ch = 0, 0, W, H
    else:
        x0, y0, cw, ch = crop
        ok = ok & (pr["mu"] > x0 - pr["rad"]) & (pr["mu"] < x0 + cw + pr["rad"])
        ok = ok & (pr["mv"] > y0 - pr["rad"]) & (pr["mv"] < y0 + ch + pr["rad"])

    # 球谐颜色
    if cam_center is None:
        Rv, tv = viewmat[:3, :3], viewmat[:3, 3]
        cam_center = -Rv.T @ tv
    dirs = means - cam_center[None]
    dirs = dirs / (dirs.norm(dim=1, keepdim=True) + EPS)
    basis = sh_basis(dirs, shdeg)                     # (N,K)
    sh = torch.cat([sh0, shN], 1)[:, :basis.shape[1]]  # (N,K,3)
    colors = (sh * basis[:, :, None]).sum(1) + 0.5    # (N,3)

    idx = torch.nonzero(ok, as_tuple=False).squeeze(1)
    n = idx.numel()
    if n == 0:
        return (torch.zeros(ch, cw, 3, device=means.device), torch.zeros(ch, cw, device=means.device), {})

    mu, mv = pr["mu"][idx], pr["mv"][idx]
    cov2 = pr["cov2"][idx]
    det = pr["det"][idx]
    ca = cov2[:, 1, 1] / det
    cb = -cov2[:, 0, 1] / det
    cc = cov2[:, 0, 0] / det
    rad = pr["rad"][idx]
    dep = pr["depth"][idx]
    opac = torch.sigmoid(logit_opac[idx])
    col = colors[idx]

    rim = torch.ceil(rad).clamp(min=1, max=max_fp).int()
    w = 2 * rim + 1
    cnt = w * w
    Tp = int(cnt.sum().item())
    if max_alpha_pairs is not None and Tp > max_alpha_pairs:
        perm = torch.randperm(n, device=means.device)
        idx = idx[perm]
        mu, mv, ca, cb, cc, dep = mu[perm], mv[perm], ca[perm], cb[perm], cc[perm], dep[perm]
        rad, opac, col = rad[perm], opac[perm], col[perm]
        n = max(1000, int(n * max_alpha_pairs / Tp))
        idx, mu, mv, ca, cb, cc, dep, rad, opac, col = (
            idx[:n], mu[:n], mv[:n], ca[:n], cb[:n], cc[:n], dep[:n], rad[:n], opac[:n], col[:n])
        rim = torch.ceil(rad).clamp(min=1, max=max_fp).int()
        w = 2 * rim + 1
        cnt = w * w
        Tp = int(cnt.sum().item())

    start = torch.cumsum(cnt, 0) - cnt
    gid = torch.repeat_interleave(torch.arange(n, device=means.device), cnt)
    local = torch.arange(Tp, device=means.device) - torch.repeat_interleave(start, cnt)
    wr = torch.repeat_interleave(w, cnt)
    rr = torch.repeat_interleave(rim, cnt)
    dx = (local % wr) - rr
    dy = (local // wr) - rr

    px = torch.floor(mu[gid]).long() + dx
    py = torch.floor(mv[gid]).long() + dy
    inb = (px >= x0) & (px < x0 + cw) & (py >= y0) & (py < y0 + ch)
    dxf = (px.to(mu.dtype) + 0.5) - mu[gid]
    dyf = (py.to(mu.dtype) + 0.5) - mv[gid]
    power = -0.5 * (ca[gid] * dxf * dxf + 2 * cb[gid] * dxf * dyf + cc[gid] * dyf * dyf)
    alpha = opac[gid] * torch.exp(torch.clamp(power, max=0.0)) * inb.to(mu.dtype)
    keep = alpha > (1.0 / 255.0)
    if keep.sum() == 0:
        return (torch.zeros(ch, cw, 3, device=means.device), torch.zeros(ch, cw, device=means.device), {})

    gid, px, py, alpha, dep = gid[keep], px[keep], py[keep], alpha[keep], dep[gid[keep]]
    colp = col[gid]

    pix = (py - y0) * cw + (px - x0)
    key = (pix.to(torch.int64) << 24) | torch.clamp((dep * 1e5), 0, 16777215).to(torch.int64)
    order = torch.sort(key).indices
    pix, alpha, colp, gid, dep = pix[order], alpha[order], colp[order], gid[order], dep[order]

    idxA = torch.arange(alpha.numel(), device=alpha.device)
    is_start = torch.ones_like(pix, dtype=torch.bool)
    is_start[1:] = pix[1:] != pix[:-1]
    seg_start = torch.cummax(torch.where(is_start, idxA, torch.zeros_like(idxA)), 0).values
    log_a = torch.log1p(-alpha.clamp(max=0.9999))
    cum = torch.cumsum(log_a, 0)            # 含当前元素的前缀和
    exc = cum - log_a                       # 不含当前元素
    prev = torch.where(seg_start > 0, cum[(seg_start - 1).clamp(min=0)], torch.zeros_like(cum))
    Trel = torch.exp(exc - prev)            # 段内、当前元素之前的透射率
    wgt = alpha * Trel

    img = torch.zeros(ch * cw, 3, device=means.device, dtype=colp.dtype)
    img.index_add_(0, pix, colp * wgt[:, None])
    al = torch.zeros(ch * cw, device=means.device, dtype=alpha.dtype)
    al.index_add_(0, pix, wgt)
    return img.view(ch, cw, 3), al.view(ch, cw), {"means2d": pr["means2d"], "sel": idx, "rad": pr["rad"],
                                                 "n": n, "pairs": int(Tp)}
