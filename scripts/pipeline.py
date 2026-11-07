# -*- coding: utf-8 -*-
r"""mp4 -> 3DGS 泼溅网页 全流程（Spirula Studio）

用法:
  python pipeline.py --video d:\建模\VID_xxx.mp4 --name v959 --frames 160
  python pipeline.py --name v959 --steps sfm,dataset          # 只重跑某几步

步骤:
  extract  从 mp4 均匀抽 N 张 4K 全景帧          -> recon\<name>\images\e_0000.jpg
  sfm      Spirula sfm auto（全景自动转透视）    -> recon\<name>\ws\sparse\0
  dataset  组织 ds 目录（images/sparse 链接）    -> tools\spirula\ds_<name>
  train    Spirula train 3dgs（run2 同参）       -> tools\spirula\out\run_<name>
  export   ckpt -> 单文件网页（官方 WebGL2 渲染器，手机摇杆已内置）
           out\run_<name>\splat_view.html        电脑版（全量高斯）
           out\run_<name>\splat_view_mobile.html 手机版（约 120 万高斯）
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = os.path.join(ROOT, "tools", "spirula", "app", "spirula.exe")
SPIRULA_OUT = os.path.join(ROOT, "tools", "spirula")

ap = argparse.ArgumentParser()
ap.add_argument("--video", default=r"d:\建模\VID_20261024_115439_00_959.mp4")
ap.add_argument("--name", default="v959", help="任务名，决定目录与输出名")
ap.add_argument("--frames", type=int, default=160, help="抽帧数量（run2 为 160）")
ap.add_argument("--width", type=int, default=2880)
ap.add_argument("--height", type=int, default=1440)
ap.add_argument("--iters", type=int, default=9000)
ap.add_argument("--cap", type=int, default=1200000)
ap.add_argument("--sh", type=int, default=3)
ap.add_argument("--resdiv", type=int, default=2)
ap.add_argument("--quality", default="high", choices=["low", "medium", "high", "ultra"])
ap.add_argument("--dtype", default="video", choices=["individual", "video", "internet"],
                help="拍摄类型：连续视频用 video（顺序匹配+回环闭合）")
ap.add_argument("--overlap", type=int, default=15, help="顺序匹配的相邻帧数")
ap.add_argument("--camera_model", default="equirectangular", help="镜头模型；360 全景用 equirectangular")
ap.add_argument("--floater", default="strong", choices=["off", "mild", "strong"], help="漂浮团块清理强度")
ap.add_argument("--runname", default="", help="输出目录名，默认 run_<name>")
ap.add_argument("--steps", default="extract,sfm,dataset,train,export")
ap.add_argument("--export_sh", type=int, default=1, choices=[0, 1, 2, 3],
                help="导出网页的 SH 阶数（1 阶约 100 MB，3 阶约 260 MB）")
ap.add_argument("--mobile_splats", type=int, default=1200000,
                help="手机版保留的高斯数（按不透明度挑选）")
a = ap.parse_args()

if not os.path.exists(EXE):
    sys.exit("找不到 Spirula: " + EXE)

NAME = a.name
REC = os.path.join(ROOT, "recon", NAME)
IMG = os.path.join(REC, "images")
WS = os.path.join(REC, "ws")
DS = os.path.join(SPIRULA_OUT, "ds_" + NAME)
RUN = a.runname or ("run_" + NAME)
OUT = os.path.join(SPIRULA_OUT, "out", RUN)


LOG = None


def logline(s=""):
    """同时写终端与日志文件，逐行立即刷出"""
    print(s, flush=True)
    if LOG is not None:
        LOG.write(str(s) + "\n")
        LOG.flush()


def run(cmd, allow_fail=False, stop_on=None):
    """流式执行外部命令：逐行转发 stdout/stderr，记录耗时与退出码。

    stop_on: 看到该关键字表示任务已完成（Spirula 训练结束后会停在查看器循环里
    等待 Ctrl-C），此时主动结束子进程，避免管线永久挂起。
    """
    logline(">> " + " ".join(cmd))
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace", bufsize=1)
    stopped = False
    keys = stop_on if isinstance(stop_on, (list, tuple)) else ((stop_on,) if stop_on else ())
    for line in p.stdout:
        logline(line.rstrip())
        hit = next((k for k in keys if k in line), None)
        if hit:
            stopped = True
            logline("检测到 '%s'：任务已完成，结束查看器进程" % hit)
            p.terminate()
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
            break
    rc = p.wait() if not stopped else 0
    logline("<< exit=%d%s  %.1fs" % (rc, " (任务完成后主动结束)" if stopped else "", time.time() - t0))
    if rc != 0 and not allow_fail:
        sys.exit(rc)


def junction(link, target):
    # 先校验目标，再动旧链接——否则目标缺失时会把原本完好的链接删掉
    if not os.path.exists(target):
        sys.exit("链接目标不存在：" + target)
    if os.path.lexists(link):
        is_j = os.path.islink(link) or getattr(os.path, "isjunction", lambda p: False)(link)
        if not is_j:
            sys.exit("已存在且不是链接，拒绝覆盖：" + link)
        r = subprocess.run(["cmd", "/c", "rmdir", link], capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit("删除旧链接失败：%s\n%s" % (link, r.stderr))
    os.makedirs(os.path.dirname(link), exist_ok=True)
    r = subprocess.run(["cmd", "/c", "mklink", "/J", link, target], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("创建链接失败：%s -> %s\n%s" % (link, target, r.stderr))
    logline("link %s -> %s" % (link, target))


def step_extract():
    import cv2
    if not os.path.exists(a.video):
        sys.exit("视频不存在：" + a.video)
    os.makedirs(IMG, exist_ok=True)
    for f in os.listdir(IMG):           # 换帧数时清掉旧帧，避免两套时间点混在一起
        if f.endswith(".jpg"):
            os.remove(os.path.join(IMG, f))
    cap = cv2.VideoCapture(a.video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if n <= 0:
        cap.release()
        sys.exit("无法读取视频帧数：" + a.video)
    logline("video frames=%d fps=%.2f dur=%.1fs" % (n, fps, n / (fps or 1)))
    want = min(a.frames, n)
    if want != a.frames:
        logline("请求 %d 帧，视频只有 %d 帧，按 %d 帧抽取" % (a.frames, n, want))
    want_set = set(int(round(i * (n - 1) / max(1, want - 1))) for i in range(want))
    if len(want_set) != want:
        logline("警告：抽帧去重后只有 %d 帧（请求 %d）" % (len(want_set), want))
    saved = 0
    # 顺序解码（seek 每次都从头解码，5300 帧随机访问比顺序慢 10 倍以上）
    for fi in range(n):
        ok = cap.grab()
        if not ok:
            break
        if fi not in want_set:
            continue
        ok, im = cap.retrieve()
        if not ok:
            continue
        if (im.shape[1], im.shape[0]) != (a.width, a.height):
            im = cv2.resize(im, (a.width, a.height), interpolation=cv2.INTER_AREA)
        p = os.path.join(IMG, "e_%04d.jpg" % saved)
        ok, buf = cv2.imencode(".jpg", im, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            sys.exit("encode failed at frame %d" % fi)
        buf.tofile(p)   # cv2.imwrite 不支持中文路径
        saved += 1
        if saved % 50 == 0:
            logline("  %d/%d" % (saved, want))
    cap.release()
    if saved == 0:
        sys.exit("一帧都没抽出来：" + a.video)
    logline("extract -> %s  %d frames" % (IMG, saved))


SFM_QUALITY = {"ultra": "extreme"}      # sfm 最高档叫 extreme，训练叫 ultra


def frame_sig():
    """当前帧集合的指纹：帧数 + 总字节数 + 视频文件信息"""
    fs = sorted(f for f in os.listdir(IMG) if f.endswith(".jpg"))
    st = os.stat(a.video)
    return {"n": len(fs),
            "bytes": sum(os.path.getsize(os.path.join(IMG, f)) for f in fs),
            "video": os.path.abspath(a.video), "vbytes": st.st_size, "vmtime": int(st.st_mtime)}


def step_sfm():
    if not os.path.isdir(IMG) or not any(f.endswith(".jpg") for f in os.listdir(IMG)):
        sys.exit("抽帧目录为空：" + IMG)
    sig = frame_sig()
    sigp = os.path.join(REC, "sfm_sig.json")
    has_sfm = os.path.exists(os.path.join(WS, "sparse", "0", "images.bin"))
    if has_sfm:
        old = None
        if os.path.exists(sigp):
            try:
                old = json.load(open(sigp, encoding="utf-8"))
            except Exception:
                old = None
        if old is not None and old != sig:
            sys.exit("已有 SfM 结果与当前帧集合/视频不一致：\n  旧 %s\n  新 %s\n请先删除 %s 再重跑" % (old, sig, WS))
        logline("发现已有 SfM 结果，沿用现有 sparse/0（若想重新解算，先删 %s）" % WS)
    os.makedirs(WS, exist_ok=True)
    cmd = [EXE, "sfm", "auto", IMG, "-o", WS,
           "--quality", SFM_QUALITY.get(a.quality, a.quality),
           "--data-type", a.dtype, "--overlap", str(a.overlap),
           "--no-masks"]
    if a.camera_model:
        cmd += ["--camera-model", a.camera_model]
    run(cmd)
    if not os.path.exists(os.path.join(WS, "sparse", "0", "images.bin")):
        sys.exit("SfM 没有产出 sparse/0/images.bin：" + WS)
    with open(sigp, "w", encoding="utf-8") as f:
        json.dump(sig, f, ensure_ascii=False)


def step_dataset():
    junction(os.path.join(DS, "images"), IMG)
    junction(os.path.join(DS, "sparse", "0"), os.path.join(WS, "sparse", "0"))
    logline("dataset -> %s" % DS)


def step_train():
    if not os.path.isdir(os.path.join(DS, "images")):
        sys.exit("数据集未就绪，请先跑 dataset 步：" + DS)
    if not os.path.exists(os.path.join(DS, "sparse", "0", "images.bin")):
        sys.exit("数据集缺 sparse/0：" + DS)
    if os.path.isdir(OUT) and os.listdir(OUT):
        sys.exit("输出目录已存在训练结果：%s\n换 --runname，或确认不要后手动删除该目录" % OUT)
    os.makedirs(os.path.join(SPIRULA_OUT, "out"), exist_ok=True)
    run([EXE, "train", "3dgs", "--data", DS,
         "--output-dir-prefix", os.path.join(SPIRULA_OUT, "out"),
         "--output-dir-name", RUN,
         "--quality", a.quality,
         "--colmap-recon-dir", "sparse/0",
         "--floater-suppression", a.floater,
         "--num-iterations", str(a.iters), "--cap-max", str(a.cap),
         "--sh-degree", str(a.sh), "--train-resolution-divisor", str(a.resdiv)],
        stop_on="查看器仍在运行")


def ckpt_step(name):
    try:
        return int(name.split("-")[1].split(".")[0])
    except Exception:
        return -1


def step_export():
    src = os.path.join(WS, "sparse", "0")
    if not os.path.exists(os.path.join(src, "images.bin")):
        sys.exit("缺少 SfM 结果：" + src)
    if not os.path.isdir(OUT):
        sys.exit("训练输出目录不存在：" + OUT)
    ck = os.path.join(OUT, "step-%09d.ckpt" % a.iters)
    if not os.path.isdir(ck):
        cands = [f for f in os.listdir(OUT)
                 if f.endswith(".ckpt") and os.path.isdir(os.path.join(OUT, f))]
        if not cands:
            sys.exit("没有找到 checkpoint：" + OUT)
        ck = os.path.join(OUT, max(cands, key=ckpt_step))
        logline("使用 checkpoint：%s" % ck)
    ply = os.path.join(ck, "splat.ply")
    if not os.path.exists(ply):
        sys.exit("缺 splat.ply: " + ply)
    gen = os.path.join(ROOT, "scripts", "splat_ply_to_web.py")
    tpl = os.path.join(ROOT, "scripts", "splat_tpl2.html")
    base = [sys.executable, gen, "--ply", ply, "--sparse", src,
            "--tmpl", tpl, "--sh_degree", str(a.export_sh)]
    # 电脑版：全量高斯，2160 宽上限
    run(base + ["--out", os.path.join(OUT, "splat_view.html"),
                "--maxsplats", "0", "--res_cap", "2160", "--dpr_cap", "2"])
    # 手机版：裁剪高斯 + 1440 宽上限 + 大摇杆
    run(base + ["--out", os.path.join(OUT, "splat_view_mobile.html"),
                "--maxsplats", str(a.mobile_splats), "--res_cap", "1440",
                "--dpr_cap", "1.5", "--mobile"])


STEPS = {"extract": step_extract, "sfm": step_sfm, "dataset": step_dataset,
         "train": step_train, "export": step_export}

os.makedirs(REC, exist_ok=True)
LOG = open(os.path.join(REC, "pipeline.log"), "a", encoding="utf-8")
logline("\n==== %s  steps=%s  %s ====" % (NAME, a.steps, time.strftime("%Y-%m-%d %H:%M:%S")))
logline("video=%s frames=%d iters=%d cap=%d sh=%d resdiv=%d"
        % (a.video, a.frames, a.iters, a.cap, a.sh, a.resdiv))

for s in a.steps.split(","):
    s = s.strip()
    if not s:
        continue
    if s not in STEPS:
        sys.exit("unknown step: " + s)
    logline("\n======== step: %s  %s ========" % (s, time.strftime("%H:%M:%S")))
    t0 = time.time()
    STEPS[s]()
    logline("======== done: %s  %.1fs ========" % (s, time.time() - t0))
LOG.close()
print("\ndone.")
