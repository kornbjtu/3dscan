#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Insta360 X5 .insv 加速流水线

把相机原片配对 (VID_YYYYMMDD_HHMMSS_00_NNN.insv + LRV_YYYYMMDD_HHMMSS_01_NNN.lrv)
处理成相机相册可识别、显示正确时长的成片:

  1. 画面加速到 --target 秒(默认 120s), 帧率固定 --fps(默认 30), insv 丢弃音频
  2. 重编码后的 mp4 做"盒子手术":
     - ftyp 换回相机原厂签名(长度不足时用原厂 compatible brand 补齐)
     - insv 前缀 = ftyp + free(8); lrv 前缀 = ftyp + wide(8), 保持 mdat 偏移不变
     - 从**源文件各自**取出 udta(AMBA/nail/free) 和 inst(陀螺仪) 注入新文件
     - trak 的 hdlr handler_name 改回原厂 Pascal 串 \x10INS.HVC / \x10INS.AVC / \x10INS.AAC
     - stco/co64 按 mdat 偏移差做 delta 回填(正常情况下 delta=0, 不触发)
  3. 回填 AMBA 私有元数据(相机相册读的是这里, 不是 moov):
     - 每条 42 字节记录的时长 = target * 记录自带 timescale
     - 每条记录的帧数 = target * fps
     - trailer 内音频采样数字段按比例缩放
     - inst 内的陀螺仪采样数**保持不动**(数据长度没变)
  4. ffprobe 复核产物

示例:
  python insv_pipeline.py --dir "E:\\DCIM\\Camera01" --outdir "d:\\建模\\out"
  python insv_pipeline.py --insv "d:\\建模\\VID_20261018_143310_00_899.insv"
  python insv_pipeline.py --insv <源> --enc-insv <已编码.mp4> --enc-lrv <已编码.mp4>   # 跳过编码
"""

import argparse
import json
import os
import re
import struct
import subprocess
import sys
import tempfile

# 环境里 ffmpeg 不在 PATH 中, 用绝对路径调用
FFMPEG = os.environ.get("INSV_FFMPEG", r"D:\剪辑\JianyingPro\6.3.0.12011\ffmpeg.exe")
FFPROBE = os.environ.get("INSV_FFPROBE", r"D:\Trae CN\resources\app\bin\ffprobe.exe")

NAME_RE = re.compile(r"^(VID|LRV)_(\d{8})_(\d{6})_(\d{2})_(\d{3})\.(insv|lrv)$", re.I)

# 原厂 hdlr handler_name (Pascal 串: 长度字节 0x10 + 16 字节)
HDL_R_NAMES = {
    (b"vide", b"hvc1"): b"\x10INS.HVC" + b"\x00" * 9,
    (b"vide", b"hev1"): b"\x10INS.HVC" + b"\x00" * 9,
    (b"vide", b"avc1"): b"\x10INS.AVC" + b"\x00" * 9,
    (b"soun", b"mp4a"): b"\x10INS.AAC" + b"\x00" * 9,
}


# ---------------------------------------------------------------- 基础工具

def sh(cmd, quiet=True):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = p.stdout.decode("utf-8", "replace")
    if p.returncode != 0:
        print(out[-4000:])
        raise RuntimeError("命令失败 rc=%s: %s" % (p.returncode, " ".join(map(str, cmd))))
    if not quiet:
        print(out[-2000:])
    return out


def probe(path):
    out = sh([FFPROBE, "-v", "error", "-print_format", "json",
              "-show_streams", "-show_format", path])
    return json.loads(out)


def video_duration(info, index=0):
    vids = [s for s in info["streams"] if s.get("codec_type") == "video"]
    if index < len(vids) and vids[index].get("duration"):
        return float(vids[index]["duration"])
    for s in info["streams"]:
        if s.get("duration"):
            return float(s["duration"])
    return float(info["format"].get("duration", 0.0))


def top_boxes(path):
    """流式解析顶层盒, 返回 [(off, size, type, hdrsize)]"""
    size = os.path.getsize(path)
    res = []
    with open(path, "rb") as f:
        off = 0
        while off + 8 <= size:
            f.seek(off)
            h = f.read(16)
            if len(h) < 8:
                break
            bsize, typ = struct.unpack(">I4s", h[:8])
            hd = 8
            if bsize == 1:
                bsize = struct.unpack(">Q", h[8:16])[0]
                hd = 16
            elif bsize == 0:
                bsize = size - off
            if bsize < 8:
                break
            res.append((off, bsize, typ.decode("latin1"), hd))
            off += bsize
    return res


def iter_boxes(buf, start, end):
    """在内存缓冲里遍历盒"""
    n = len(buf)
    end = min(end, n)
    off = start
    while off + 8 <= end:
        bsize, typ = struct.unpack_from(">I4s", buf, off)
        hd = 8
        if bsize == 1:
            bsize = struct.unpack_from(">Q", buf, off + 8)[0]
            hd = 16
        elif bsize == 0:
            bsize = end - off
        if bsize < hd or off + bsize > n:
            break
        yield off, bsize, typ, hd
        off += bsize


def find_child(buf, start, end, typ):
    for o, s, t, hd in iter_boxes(buf, start, end):
        if t == typ:
            return o, s, hd
    return None


def find_deep(buf, names, start, end):
    for name in names:
        r = find_child(buf, start, end, name)
        if not r:
            return None
        start, end = r[0] + r[2], r[0] + r[1]
    return start, end


def read_at(path, off, size):
    with open(path, "rb") as f:
        f.seek(off)
        return f.read(size)


def mkbox(typ, payload):
    return struct.pack(">I", len(payload) + 8) + typ + payload


def top_of(tops, typ):
    return next(((o, s, h) for o, s, t, h in tops if t == typ), None)


# ---------------------------------------------------------------- 盒子手术

def fit_ftyp(orig, tgt_len, enc_compat):
    """把源文件 ftyp 调成 tgt_len: 等长直接用; 变长追加原编码器的 compatible brand; 变短截断"""
    if len(orig) == tgt_len:
        return orig
    if tgt_len < len(orig):
        return orig[:tgt_len]
    if tgt_len < 20 or (tgt_len - len(orig)) % 4 != 0:
        return None
    present = [orig[i:i + 4] for i in range(16, len(orig), 4)]
    need = tgt_len - len(orig)
    add = bytearray()
    for b in reversed(enc_compat):          # 倒序取, 与已验证成片的选择一致
        if len(add) >= need:
            break
        if b not in present:
            add += b
    while len(add) < need:
        add += b"\x00\x00\x00\x00"
    return struct.pack(">I", tgt_len) + b"ftyp" + bytes(orig[8:]) + bytes(add)


def stsd_codec(buf, start, end):
    p = find_deep(buf, [b"minf", b"stbl", b"stsd"], start, end)
    if not p:
        return None
    s, e = p
    for o, s2, t, hd in iter_boxes(buf, s + 8, e):   # 8 hdr + 4 ver/flags + 4 entry_count
        return t
    return None


def fix_trak(trak):
    """把 trak 里的 hdlr handler_name 换成原厂 Pascal 串, 并同步父盒尺寸"""
    md = find_child(trak, 8, len(trak), b"mdia")
    if not md:
        return trak
    mo, ms, mhd = md
    h = find_child(trak, mo + mhd, mo + ms, b"hdlr")
    if not h:
        return trak
    ho, hs, hhd = h
    htype = trak[ho + hhd + 8: ho + hhd + 12]
    cc = stsd_codec(trak, mo + mhd, mo + ms)
    name = HDL_R_NAMES.get((htype, cc))
    if name is None and htype == b"soun":
        name = HDL_R_NAMES[(b"soun", b"mp4a")]
    if name is None:
        return trak
    new_h = mkbox(b"hdlr", trak[ho + hhd: ho + hhd + 24] + name)
    body = trak[mo + mhd: ho] + new_h + trak[ho + hs: mo + ms]
    return mkbox(b"trak", trak[8: mo] + mkbox(b"mdia", body) + trak[mo + ms:])


def patch_chunk_offsets(buf, delta):
    """moov 缓冲内所有 stco/co64 条目 += delta (只改数值不改尺寸)"""
    def walk(start, end):
        for o, s, t, hd in iter_boxes(buf, start, end):
            if t in (b"trak", b"mdia", b"minf", b"stbl"):
                walk(o + hd, o + s)
            elif t in (b"stco", b"co64"):
                n = struct.unpack_from(">I", buf, o + hd + 4)[0]
                base = o + hd + 8
                w = 4 if t == b"stco" else 8
                fmt = ">I" if t == b"stco" else ">Q"
                for i in range(n):
                    p = base + w * i
                    v = struct.unpack_from(fmt, buf, p)[0]
                    struct.pack_into(fmt, buf, p, v + delta)
    walk(8, len(buf))


def build_container(src, enc, out, is_lrv):
    """把重编码 mp4 与源文件的私有元数据拼成相机可识别的成片"""
    src_top = top_boxes(src)
    enc_top = top_boxes(enc)

    ft = top_of(src_top, "ftyp")
    mo = top_of(src_top, "moov")
    inst = top_of(src_top, "inst")
    if not (ft and mo and inst):
        raise RuntimeError("源文件缺少 ftyp/moov/inst: %s" % src)
    src_moov = read_at(src, mo[0], mo[1])
    ud = find_child(src_moov, 8, len(src_moov), b"udta")
    if not ud:
        raise RuntimeError("源文件 moov 内没有 udta(AMBA/nail/free): %s" % src)
    orig_udta = src_moov[ud[0]: ud[0] + ud[1]]
    orig_ftyp = read_at(src, ft[0], ft[1])
    orig_inst = read_at(src, inst[0], inst[1])

    enc_mdat = top_of(enc_top, "mdat")
    enc_moov_box = top_of(enc_top, "moov")
    enc_ftyp_box = top_of(enc_top, "ftyp")
    if not (enc_mdat and enc_moov_box):
        raise RuntimeError("编码产物缺少 mdat/moov: %s" % enc)
    enc_moov = read_at(enc, enc_moov_box[0], enc_moov_box[1])
    enc_ftyp = read_at(enc, enc_ftyp_box[0], enc_ftyp_box[1]) if enc_ftyp_box else b""
    enc_compat = [enc_ftyp[i:i + 4] for i in range(16, len(enc_ftyp), 4)]

    # 前缀: 让 mdat 落在与编码产物相同的位置 (通常 delta=0)
    tgt_ftyp = enc_mdat[0] - 8
    ftyp = fit_ftyp(orig_ftyp, tgt_ftyp, enc_compat) or orig_ftyp
    prefix = ftyp + mkbox(b"wide" if is_lrv else b"free", b"")
    delta = len(prefix) - enc_mdat[0]

    # moov: 编码产物的子盒 + 源文件原厂 udta(放最后), hdlr 改原厂名
    kids = []
    for o, s, t, hd in iter_boxes(enc_moov, 8, len(enc_moov)):
        if t in (b"udta", b"free", b"skip"):
            continue
        b = enc_moov[o:o + s]
        kids.append(fix_trak(b) if t == b"trak" else b)
    new_moov = bytearray(mkbox(b"moov", b"".join(kids) + orig_udta))
    if delta:
        patch_chunk_offsets(new_moov, delta)

    with open(out, "wb") as dst:
        dst.write(prefix)
        with open(enc, "rb") as ef:                    # 流式拷 mdat, 不占内存
            ef.seek(enc_mdat[0])
            left = enc_mdat[1]
            while left > 0:
                chunk = ef.read(min(1 << 22, left))
                if not chunk:
                    break
                dst.write(chunk)
                left -= len(chunk)
        dst.write(bytes(new_moov))
        dst.write(orig_inst)

    return {"prefix_len": len(prefix), "delta": delta,
            "mdat": enc_mdat[1], "moov": len(new_moov), "inst": len(orig_inst)}


def patch_amba(path, target, src_dur, fps, a_rate=None, log=print):
    """回填相机相册读取的 AMBA 私有元数据"""

    tops = top_boxes(path)
    mo = top_of(tops, "moov")
    buf = bytearray(read_at(path, mo[0], mo[1]))
    ud = find_child(buf, 8, len(buf), b"udta")
    am = find_child(buf, ud[0] + ud[2], ud[0] + ud[1], b"AMBA")
    if not am:
        log("    ! 未找到 AMBA, 跳过")
        return None
    payload_off = mo[0] + am[0] + am[2]
    size = am[1] - am[2]
    with open(path, "r+b") as f:
        f.seek(payload_off)
        payload = bytearray(f.read(size))
        pos, recs = 16, []
        while pos < len(payload) and payload[pos] == 0x2A:
            recs.append(pos)
            pos += payload[pos]
        trailer = pos
        info = {"records": len(recs), "trailer": trailer, "changes": []}
        for r in recs:
            ts = struct.unpack_from("<H", payload, r + 1)[0]
            old_d = struct.unpack_from("<I", payload, r + 0x11)[0]
            old_f = struct.unpack_from("<I", payload, r + 0x19)[0]
            new_d = int(round(target * ts))
            new_f = int(round(target * fps))
            struct.pack_into("<I", payload, r + 0x11, new_d)
            struct.pack_into("<I", payload, r + 0x19, new_f)
            info["changes"].append(("rec@%#x ts=%d" % (r, ts), old_d, new_d, old_f, new_f))
        if trailer + 0x15 <= len(payload):
            old_a = struct.unpack_from("<I", payload, trailer + 0x11)[0]
            if old_a:
                # 与已验证成片一致: 音频采样数 = 目标时长 * 采样率 (缺采样率时按比例缩放)
                if a_rate:
                    new_a = int(round(target * a_rate))
                elif src_dur > 0:
                    new_a = int(round(old_a * target / src_dur))
                else:
                    new_a = old_a
                struct.pack_into("<I", payload, trailer + 0x11, new_a)
                info["changes"].append(("audio@trailer+0x11", old_a, new_a, None, None))
        f.seek(payload_off)
        f.write(payload)
    return info


# ---------------------------------------------------------------- 编码

def atempo_chain(f):
    parts = []
    while f > 2.0:
        parts.append(2.0)
        f /= 2.0
    while f < 0.5:
        parts.append(0.5)
        f /= 0.5
    parts.append(round(f, 8))
    return ",".join("atempo=%s" % p for p in parts)


def enc_insv(src, dst, factor, fps, bitrate):
    flt = ("[0:v:0]setpts=PTS/{f},fps={fps},scale=in_range=full:out_range=full,format=nv12[v0];"
           "[0:v:1]setpts=PTS/{f},fps={fps},scale=in_range=full:out_range=full,format=nv12[v1]"
           ).format(f=factor, fps=fps)
    sh([FFMPEG, "-hide_banner", "-y", "-hwaccel", "cuda", "-i", src,
        "-filter_complex", flt, "-map", "[v0]", "-map", "[v1]", "-an",
        "-c:v", "hevc_nvenc", "-b:v", bitrate, "-color_range", "pc", "-tag:v", "hvc1",
        "-f", "mp4", dst])


def enc_lrv(src, dst, factor, fps, info, keep_audio, bitrate, abr):
    aud = [s for s in info["streams"] if s.get("codec_type") == "audio"]
    parts = ["[0:v]setpts=PTS/{f},fps={fps},scale=in_range=full:out_range=full,format=yuv420p[v]"
             .format(f=factor, fps=fps)]
    maps = ["-map", "[v]"]
    if keep_audio and aud:
        a = aud[0]
        parts.append("[0:a]%s[a]" % atempo_chain(factor))
        maps += ["-map", "[a]", "-c:a", "aac",
                 "-ac", str(a.get("channels", 2)),
                 "-ar", str(a.get("sample_rate", 48000)),
                 "-b:a", abr]
    sh([FFMPEG, "-hide_banner", "-y", "-hwaccel", "cuda", "-i", src,
        "-filter_complex", ";".join(parts)] + maps +
       ["-c:v", "h264_nvenc", "-profile:v", "high", "-b:v", bitrate,
        "-color_range", "pc", "-tag:v", "avc1", "-f", "mp4", dst])


# ---------------------------------------------------------------- 单文件流程

def verify(path, label):
    info = probe(path)
    vids = [s for s in info["streams"] if s.get("codec_type") == "video"]
    out = []
    for i, s in enumerate(vids):
        out.append("v%d %s %sx%s %s fps %s帧 %ss" % (
            i, s.get("codec_name"), s.get("width"), s.get("height"),
            s.get("r_frame_rate"), s.get("nb_frames"), s.get("duration")))
    print("    [%s] %s  容器=%ss  大小=%.2fGB" % (
        label, " | ".join(out), info["format"].get("duration"),
        os.path.getsize(path) / 1024 ** 3))
    return info


def lrv_name_for(insv_path):
    d, b = os.path.split(insv_path)
    m = NAME_RE.match(b)
    if not m:
        return None
    return os.path.join(d, "LRV_%s_%s_01_%s.lrv" % (m.group(2), m.group(3), m.group(5)))


def out_name(src_path, kind, seq_digits, seq=None):
    d, b = os.path.split(src_path)
    m = NAME_RE.match(b)
    if not m:
        raise RuntimeError("文件名不符合 Insta360 规范: %s" % b)
    num = seq_digits if seq_digits is not None else (seq if seq is not None else m.group(5))
    pre = "VID" if kind == "insv" else "LRV"
    mid = "00" if kind == "insv" else "01"
    return "%s_%s_%s_%s_%s.%s" % (pre, m.group(2), m.group(3), mid, num, kind)


def process(args, insv, lrv, seq=None):
    base = os.path.basename(insv)
    print("==", base)
    src_info = probe(insv)
    vids = [s for s in src_info["streams"] if s.get("codec_type") == "video"]
    if len(vids) < 2:
        print("    ! 源 insv 视频流只有 %d 条, 期望 2 条(前后镜头), 跳过" % len(vids))
        return False
    src_dur = video_duration(src_info, 0)
    factor = args.factor if args.factor else src_dur / float(args.target)
    if args.no_slow and factor < 1.0:
        factor = 1.0
    eff_dur = src_dur / factor                     # 实际成片时长(可能不等于 --target)
    auds = [s for s in src_info["streams"] if s.get("codec_type") == "audio"]
    a_rate = int(auds[0]["sample_rate"]) if auds and auds[0].get("sample_rate") else 48000
    print("    源时长 %.3fs -> 成片 %.3fs  变速系数 %.8f" % (src_dur, eff_dur, factor))

    if not lrv:
        lrv = lrv_name_for(insv)
    if not lrv or not os.path.exists(lrv):
        print("    ! 找不到配对的 LRV: %s -- 相机不会识别, 跳过" % lrv)
        return False
    lrv_info = probe(lrv)

    d, b = os.path.split(insv)
    lrv_base = os.path.basename(lrv)
    n_insv = out_name(b, "insv", args.seq_digits, seq)
    n_lrv = out_name(lrv_base, "lrv", args.seq_digits, seq)
    os.makedirs(args.outdir, exist_ok=True)
    out_insv = os.path.join(args.outdir, n_insv)
    out_lrv = os.path.join(args.outdir, n_lrv)
    if args.dry_run:
        print("    [dry-run] 将输出 %s / %s" % (out_insv, out_lrv))
        return True
    for p in (out_insv, out_lrv):
        if os.path.exists(p) and not args.force:
            print("    ! 已存在(加 --force 覆盖): %s" % p)
            return False

    tmpdir = tempfile.gettempdir()
    tmp_insv = args.enc_insv or os.path.join(tmpdir, "insv_%s_accel.mp4" % base)
    tmp_lrv = args.enc_lrv or os.path.join(tmpdir, "lrv_%s_accel.mp4" % lrv_base)

    if args.enc_insv and args.enc_lrv:
        print("    复用已编码中间产物, 跳过 ffmpeg")
    else:
        if not args.enc_insv:
            print("    编码 insv ...")
            enc_insv(insv, tmp_insv, factor, args.fps, args.v_bitrate)
        if not args.enc_lrv:
            print("    编码 lrv ...")
            enc_lrv(lrv, tmp_lrv, factor, args.fps, lrv_info,
                    args.keep_lrv_audio, args.lrv_bitrate, args.a_bitrate)

    r1 = build_container(insv, tmp_insv, out_insv, is_lrv=False)
    print("    重建 insv: ftyp+free=%dB delta=%d mdat=%d moov=%d inst=%d"
          % (r1["prefix_len"], r1["delta"], r1["mdat"], r1["moov"], r1["inst"]))
    r2 = build_container(lrv, tmp_lrv, out_lrv, is_lrv=True)
    print("    重建 lrv : ftyp+wide=%dB delta=%d mdat=%d moov=%d inst=%d"
          % (r2["prefix_len"], r2["delta"], r2["mdat"], r2["moov"], r2["inst"]))

    # 中间产物体积很大(每个可达数 GB), 重建成功后默认清掉, 避免堆满 %TEMP%
    if not args.keep_temp:
        for p in ((None if args.enc_insv else tmp_insv),
                  (None if args.enc_lrv else tmp_lrv)):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

    for p in (out_insv, out_lrv):
        info = patch_amba(p, eff_dur, src_dur, args.fps, a_rate)
        if info:
            print("    AMBA %s: %d 条记录, trailer@%#x" % (os.path.basename(p), info["records"], info["trailer"]))
            for c in info["changes"]:
                print("      %-24s %s -> %s%s" % (c[0], c[1], c[2],
                                                  "" if c[3] is None else "   frames %s -> %s" % (c[3], c[4])))

    if not args.no_verify:
        verify(out_insv, "insv")
        verify(out_lrv, "lrv")
    print("    完成 -> %s" % out_insv)
    return True


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Insta360 X5 .insv 加速流水线")
    ap.add_argument("--insv", action="append", default=[], help="源 insv 路径(可重复)")
    ap.add_argument("--dir", action="append", default=[], help="扫描该目录下所有 VID_*.insv")
    ap.add_argument("--lrv", help="显式指定 LRV(仅单文件时有效)")
    ap.add_argument("--outdir", default=None, help="输出目录, 默认 <源目录>\\insv_out")
    ap.add_argument("--target", type=float, default=120.0, help="目标时长秒, 默认 120")
    ap.add_argument("--fps", type=float, default=30.0, help="输出帧率, 默认 30")
    ap.add_argument("--factor", type=float, default=None, help="直接指定变速系数(覆盖 --target)")
    ap.add_argument("--no-slow", action="store_true",
                    help="变速系数下限为 1.0: 源片短于 --target 时不做慢放, 成片保持源时长")
    ap.add_argument("--seq-digits", default=None, help="强制输出文件名 3 位序号, 如 900 (批量时所有文件同名, 慎用)")
    ap.add_argument("--seq-start", type=int, default=None,
                    help="批量时从该序号起给每个成片顺序编号(与 --seq-digits 互斥), 如 901")
    ap.add_argument("--v-bitrate", default="60M", help="insv 视频码率, 默认 60M")
    ap.add_argument("--lrv-bitrate", default="8M", help="LRV 视频码率, 默认 8M")
    ap.add_argument("--a-bitrate", default="256k", help="LRV 音频码率, 默认 256k")
    ap.add_argument("--drop-lrv-audio", dest="keep_lrv_audio", action="store_false",
                    help="LRV 也丢弃音频(默认保留变速音频)")
    ap.add_argument("--enc-insv", help="复用已编码的 insv mp4(跳过编码, 需同时给 --enc-lrv)")
    ap.add_argument("--enc-lrv", help="复用已编码的 lrv mp4")
    ap.add_argument("--force", action="store_true", help="覆盖已存在的输出")
    ap.add_argument("--keep-temp", action="store_true",
                    help="保留 %TEMP% 下的中间编码产物(默认重建成功后删除)")
    ap.add_argument("--dry-run", action="store_true", help="只探测与计算, 不编码不写文件")
    ap.add_argument("--no-verify", action="store_true", help="跳过 ffprobe 复核")
    args = ap.parse_args()

    for tool in (FFMPEG, FFPROBE):
        if not os.path.exists(tool):
            print("! 找不到工具: %s (可用环境变量 INSV_FFMPEG / INSV_FFPROBE 覆盖)" % tool)
            return 2

    insvs = list(args.insv)
    for d in args.dir:
        if not os.path.isdir(d):
            print("! 目录不存在: %s" % d)
            continue
        for fn in sorted(os.listdir(d)):
            if re.match(r"^VID_.*\.insv$", fn, re.I):
                insvs.append(os.path.join(d, fn))
    if not insvs:
        print("! 没有输入。用 --insv 指定文件或用 --dir 指定目录")
        return 2
    if args.lrv and len(insvs) > 1:
        print("! --lrv 只能配合单个 --insv 使用")
        return 2
    if args.seq_start is not None and args.seq_digits is not None:
        print("! --seq-start 与 --seq-digits 不能同时使用")
        return 2

    ok, fail = 0, 0
    for i, iv in enumerate(insvs):
        if not os.path.exists(iv):
            print("! 文件不存在: %s" % iv)
            fail += 1
            continue
        od = args.outdir or os.path.join(os.path.dirname(os.path.abspath(iv)), "insv_out")
        args.outdir = od
        lrv = args.lrv if (args.lrv and len(insvs) == 1) else None
        seq = None if args.seq_start is None else "%03d" % (args.seq_start + i)
        try:
            if process(args, iv, lrv, seq):
                ok += 1
            else:
                fail += 1
        except Exception as e:
            print("    ! 出错: %s" % e)
            fail += 1
    print("\n完成: 成功 %d, 失败/跳过 %d" % (ok, fail))
    print("注意: inst 盒内陀螺仪时间轴仍按原时长写入, 与目标时长不匹配;")
    print("      若相机预览防抖异常, 说明防抖依赖陀螺仪时间轴, 需另行重采样。")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
