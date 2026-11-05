"""从 3DGS splat.ply 提取全部高斯中心为纯 xyz 点云（保留 opacity 过滤）。"""
import sys
import numpy as np

src, dst = sys.argv[1], sys.argv[2]
NUM = {'float': 'f4', 'float32': 'f4', 'double': 'f8', 'uchar': 'u1', 'int': 'i4', 'uint': 'u4'}

with open(src, 'rb') as f:
    props = []
    n = 0
    fmt = None
    while True:
        ln = f.readline().decode().strip()
        if ln.startswith('format'):
            fmt = ln.split()[1]
        elif ln.startswith('element vertex'):
            n = int(ln.split()[-1])
        elif ln.startswith('property'):
            t = ln.split()
            props.append((t[2], t[1]))
        elif ln == 'end_header':
            break
    assert fmt == 'binary_little_endian', fmt
    dt = np.dtype([(p[0], '<' + NUM[p[1]]) for p in props])
    buf = np.frombuffer(f.read(dt.itemsize * n), dtype=dt, count=n)

xyz = np.stack([buf['x'], buf['y'], buf['z']], 1).astype('<f4')
keep = np.ones(len(xyz), bool)
if 'opacity' in dt.names:
    op = buf['opacity'].astype(np.float32)
    keep &= op > -3.0
xyz = xyz[keep]
print('gaussians', n, 'kept', len(xyz))

with open(dst, 'wb') as f:
    f.write(b'ply\nformat binary_little_endian 1.0\n')
    f.write(('element vertex %d\n' % len(xyz)).encode())
    f.write(b'property float x\nproperty float y\nproperty float z\nend_header\n')
    f.write(xyz.tobytes())
print('wrote', dst)
