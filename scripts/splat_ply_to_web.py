"""Spirula 3DGS splat.ply -> 单文件网页（官方 Spirula WebGL2 渲染器）

- 渲染器代码取自 Spirula Studio 开源仓库 viewer/js（GPL-3.0），构建时内联进 HTML
- 数据保持模型原生坐标系（Z 向上），与官方查看器一致
- 单文件：高斯数据 base64 内嵌，直接双击打开即可

用法:
  python splat_ply_to_web.py --ply <step-xxx.ckpt/splat.ply> --sparse <sparse/0> --out <splat_view.html>
"""
import argparse
import base64
import json
import os
import sys

import numpy as np

C0 = 0.28209479177387814
SH_COEFFS = {0: 0, 1: 3, 2: 8, 3: 15}
LIB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'webviewer')
LIB_FILES = ['linalg.js', 'colors.js', 'shaders.js', 'camera.js', 'renderer.js']


def log(s):
    print(s, flush=True)


# ---------------------------------------------------------------- ply / 相机
def ply_header(path):
    names, n, data_off = [], 0, 0
    with open(path, 'rb') as f:
        while True:
            ln = f.readline()
            data_off += len(ln)
            s = ln.decode('ascii', 'ignore').strip()
            if s.startswith('format'):
                fmt = s.split()[1]
            elif s.startswith('element vertex'):
                n = int(s.split()[-1])
            elif s.startswith('property'):
                names.append(s.split()[-1])
            elif s == 'end_header':
                break
    return names, n, data_off, fmt


def load_ply(path):
    names, n, off, fmt = ply_header(path)
    if fmt != 'binary_little_endian':
        raise SystemExit('只支持 binary_little_endian ply: ' + fmt)
    dt = np.dtype([(nm, '<f4') for nm in names])
    buf = np.memmap(path, dtype=dt, mode='r', offset=off, shape=(n,))
    log('ply: %d verts, %d props, %.1f MB' % (n, len(names), os.path.getsize(path) / 1048576))
    return buf, names


def read_images(path):
    txt = os.path.join(path, 'images.txt')
    binf = os.path.join(path, 'images.bin')
    if os.path.exists(txt) and not os.path.exists(binf):
        return _read_images_txt(txt)
    if not os.path.exists(binf):
        raise SystemExit('找不到 images.txt / images.bin: ' + path)
    imgs = []
    with open(binf, 'rb') as f:
        n = int(np.frombuffer(f.read(8), np.uint64)[0])
        for _ in range(int(n)):
            f.read(4)
            w, x, y, z = np.frombuffer(f.read(32), np.float64)   # Spirula 存 (w,x,y,z)
            t = np.frombuffer(f.read(24), np.float64)            # 相机中心（非 COLMAP 的 -R^T C）
            f.read(4)
            while True:                                          # 文件名 NULL 结尾
                c = f.read(1)
                if not c or c == b'\x00':
                    break
            npts = int(np.frombuffer(f.read(8), np.uint64)[0])
            f.seek(24 * npts, 1)
            R = np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                          [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                          [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])
            imgs.append((R, t))
    log('cameras(bin): %d' % len(imgs))
    return imgs


def _read_images_txt(path):
    imgs = []
    lines = [l.rstrip('\n') for l in open(path, encoding='utf-8')]
    k = 0
    while k < len(lines):
        ln = lines[k].strip()
        if not ln or ln.startswith('#'):
            k += 1
            continue
        f = ln.split()
        w, x, y, z = [float(v) for v in f[1:5]]
        t = np.array([float(v) for v in f[5:8]])
        R = np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                      [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                      [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])
        imgs.append((R, t))
        k += 2
    log('cameras(txt): %d' % len(imgs))
    return imgs


def camera_anchor(imgs):
    """默认机位：第一帧的真实相机位姿（拍摄起点）。

    不用轨迹中位数——多楼层场景的中位数经常落在墙体/楼板内部，开场糊成一片；
    第一帧一定在可通行的拍摄路径上，朝向也是当时真实视线，开场最稳。
    """
    C = np.stack([t for (_, t) in imgs])
    F = np.stack([R.T @ np.array([0.0, 0.0, 1.0]) for (R, _) in imgs])
    F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-12
    return C[0], F[0]


# ---------------------------------------------------------------- 编码
def align4(v):
    return (v + 3) & ~3


def h2f_array(a):
    h = np.asarray(a, dtype=np.float16)
    return h, h.astype(np.float32)


def bundle_libs():
    parts = []
    for name in LIB_FILES:
        src = open(os.path.join(LIB_DIR, name), encoding='utf-8').read()
        out = []
        for ln in src.split('\n'):
            if ln.startswith('import '):
                continue
            if ln.startswith('export '):
                ln = ln[7:]
            out.append(ln)
        parts.append('/* ---- %s ---- */\n%s' % (name, '\n'.join(out)))
    return '\n\n'.join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ply', required=True)
    ap.add_argument('--sparse', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tmpl', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'splat_tpl2.html'))
    ap.add_argument('--sh_degree', type=int, default=1, help='0/1/2/3，默认 1')
    ap.add_argument('--maxsplats', type=int, default=0, help='0 = 全部')
    ap.add_argument('--start_cam', type=int, default=-1, help='-1 = 相机轨迹中心')
    ap.add_argument('--cull', type=float, default=3.5, help='丢弃离起点超过该距离的离群高斯（米，0=关闭）')
    ap.add_argument('--res_cap', type=int, default=2160, help='画布长边像素上限（手机版建议 1440）')
    ap.add_argument('--dpr_cap', type=float, default=2.0, help='devicePixelRatio 上限（手机版建议 1.5）')
    ap.add_argument('--mobile', action='store_true', help='标记为手机版（放大摇杆、标题后缀）')
    a = ap.parse_args()

    buf, names = load_ply(a.ply)
    N0 = len(buf)
    log('总高斯 %d' % N0)

    def col(n):
        return np.asarray(buf[n], dtype=np.float32)

    pos = np.stack([col('x'), col('y'), col('z')], 1)
    op = 1.0 / (1.0 + np.exp(-col('opacity')))
    scl = np.exp(np.stack([col('scale_0'), col('scale_1'), col('scale_2')], 1))
    qw, qx, qy, qz = col('rot_0'), col('rot_1'), col('rot_2'), col('rot_3')
    qn = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qn[qn < 1e-12] = 1.0
    quat = np.stack([qx / qn, qy / qn, qz / qn, qw / qn], 1)

    if 'f_dc_0' in names:
        dc = np.stack([col('f_dc_0'), col('f_dc_1'), col('f_dc_2')], 1)
    else:
        dc = np.stack([(col('red') / 255 - 0.5) / C0,
                       (col('green') / 255 - 0.5) / C0,
                       (col('blue') / 255 - 0.5) / C0], 1)

    kf = sum(1 for n in names if n.startswith('f_rest_'))
    trained = {0: 0, 3: 1, 8: 2, 15: 3}.get(kf // 3, 3)
    deg = max(0, min(a.sh_degree, trained))
    K = SH_COEFFS[deg]
    log('SH: 训练 %d 阶 -> 导出 %d 阶（每高斯 %d 个系数）' % (trained, deg, K))

    if K:
        kf3 = kf // 3
        cols = [col('f_rest_%d' % (c * kf3 + k)) for k in range(K) for c in range(3)]
        sh = np.stack(cols, 1)
    else:
        sh = np.zeros((N0, 0), np.float32)

    if a.maxsplats and N0 > a.maxsplats:
        keep = np.argsort(-op)[:a.maxsplats]
        keep.sort()
        log('下采样 %d -> %d' % (N0, len(keep)))
        pos, op, scl, quat, dc, sh = (x[keep] for x in (pos, op, scl, quat, dc, sh))
        N = len(keep)
    else:
        N = N0
    log('导出高斯 %d' % N)

    # 起始机位
    imgs = read_images(a.sparse)
    if a.start_cam >= 0:
        i = a.start_cam % len(imgs)
        start = imgs[i][1].astype(np.float64)
        fwd = imgs[i][0].T @ np.array([0.0, 0.0, 1.0])
    else:
        start, fwd = camera_anchor(imgs)
    start = start.astype(np.float64)
    fwd = fwd / (np.linalg.norm(fwd) + 1e-12)
    log('起点 %s 朝向 %s' % (np.round(start, 4), np.round(fwd, 4)))

    if a.cull > 0:
        far = np.linalg.norm(pos - start, axis=1) > a.cull
        if far.any():
            keep = ~far
            log('丢弃离群高斯 %d 个（>%.1f m）' % (int(far.sum()), a.cull))
            pos, op, scl, quat, dc, sh = (x[keep] for x in (pos, op, scl, quat, dc, sh))
            N = len(pos)
            log('保留高斯 %d' % N)

    # Morton 重排（与官方查看器一致）：逐帧深度排序后属性纹理访问保持空间局部性，帧率提升明显
    lo = pos.min(0)
    span = np.maximum(pos.max(0) - lo, 1e-9)
    q3 = np.clip((pos - lo) * (1023.0 / span.max()), 0, 1023).astype(np.uint32)

    def spread(v):
        v = (v | (v << np.uint32(16))) & np.uint32(0x030000FF)
        v = (v | (v << np.uint32(8))) & np.uint32(0x0300F00F)
        v = (v | (v << np.uint32(4))) & np.uint32(0x030C30C3)
        v = (v | (v << np.uint32(2))) & np.uint32(0x09249249)
        return v

    code = spread(q3[:, 0]) | (spread(q3[:, 1]) << np.uint32(1)) | (spread(q3[:, 2]) << np.uint32(2))
    mo = np.argsort(code)
    pos, op, scl, quat, dc, sh = (x[mo] for x in (pos, op, scl, quat, dc, sh))
    log('Morton 重排完成')

    # 量化
    maxabs = float(np.abs(pos).max())
    step = max(0.0001, maxabs / 32000.0)
    posq = np.clip(np.round(pos / step), -32767, 32767).astype(np.int16)
    log('位置量化步长 %.4f mm（范围 ±%.2f m）' % (step * 1000, 32767 * step))

    opu8 = np.clip(np.round(op * 255), 0, 255).astype(np.uint8)
    q8 = np.clip(np.round((quat + 1.0) * 127.5), 0, 255).astype(np.uint8)

    scl_h, scl_f = h2f_array(scl)
    dc_h, _ = h2f_array(dc)

    if K:
        mx = np.abs(sh).max(1)
        ssh_h = np.asarray(mx, dtype=np.float16)
        ssh_f = ssh_h.astype(np.float32)
        inv = np.where(ssh_f > 0, 1.0 / np.maximum(ssh_f, 1e-20), 0.0)[:, None]
        sh8 = np.clip(np.round(sh * inv * 127.0), -127, 127).astype(np.int8)
    else:
        ssh_h = np.zeros(N, np.float16)
        sh8 = np.zeros((N, 0), np.int8)

    # 布局
    o = {}
    off = 0
    off = align4(off); o['pos'] = off; off += N * 3 * 2
    o['op'] = off; off += N
    off = align4(off); o['quat'] = off; off += N * 4
    off = align4(off); o['scale'] = off; off += N * 3 * 2
    o['shscale'] = off; off += N * 2
    off = align4(off); o['fdc'] = off; off += N * 3 * 2
    off = align4(off); o['sh'] = off; off += N * (K * 3)
    total = align4(off)

    payload = bytearray(total)
    payload[o['pos']:o['pos'] + N * 6] = posq.tobytes()
    payload[o['op']:o['op'] + N] = opu8.tobytes()
    payload[o['quat']:o['quat'] + N * 4] = q8.tobytes()
    payload[o['scale']:o['scale'] + N * 6] = scl_h.tobytes()
    payload[o['shscale']:o['shscale'] + N * 2] = ssh_h.tobytes()
    payload[o['fdc']:o['fdc'] + N * 6] = dc_h.tobytes()
    if K:
        payload[o['sh']:o['sh'] + N * K * 3] = sh8.tobytes()
    log('载荷 %.1f MB' % (total / 1048576))

    # 排序半径
    center = pos.mean(0)
    radius = float(np.percentile(np.linalg.norm(pos - center, axis=1), 99.9)) * 1.35
    radius = max(radius, 0.5)
    log('场景半径 %.2f m' % radius)

    meta = {
        'count': N, 'shDegree': deg, 'shRest': K, 'posStep': step,
        'offsets': o, 'start': [float(v) for v in start], 'fwd': [float(v) for v in fwd],
        'radius': radius, 'bytes': total,
        'resCap': a.res_cap, 'dprCap': a.dpr_cap, 'mobile': bool(a.mobile),
    }

    tpl = open(a.tmpl, encoding='utf-8').read()
    b64 = base64.b64encode(bytes(payload)).decode('ascii')
    libs = bundle_libs()
    html = (tpl.replace('__LIBS__', libs)
                .replace('__META__', json.dumps(meta, separators=(',', ':')))
                .replace('__DATA__', b64)
                .replace('__N__', format(N, ',')))
    with open(a.out, 'w', encoding='utf-8') as f:
        f.write(html)
    log('写出 %s  %.1f MB' % (a.out, os.path.getsize(a.out) / 1048576))


if __name__ == '__main__':
    main()
