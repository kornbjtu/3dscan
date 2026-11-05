# -*- coding: utf-8 -*-
"""给 run2 的 splat_view.html 注入手机触摸摇杆，输出 splat_view_mobile.html"""
import io, sys

SRC = r"d:\建模\tools\spirula\out\run2\splat_view.html"
DST = r"d:\建模\tools\spirula\out\run2\splat_view_mobile.html"

s = io.open(SRC, encoding="utf-8").read()
print("src bytes", len(s))


def rep(s, old, new, tag, count=1):
    n = s.count(old)
    if n != count:
        print("[FAIL]", tag, "occurrences =", n)
        sys.exit(1)
    print("[ok]", tag)
    return s.replace(old, new, count)


s = rep(s, '<meta charset="utf-8">',
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no,viewport-fit=cover">',
        'viewport')

s = rep(s, 'canvas{display:block;width:100%;height:100%;cursor:crosshair}',
        'canvas{display:block;width:100%;height:100%;cursor:crosshair;touch-action:none}', 'canvas')

CSS = '''  .hidden{display:none!important}
  #stick{position:fixed;left:26px;bottom:34px;width:136px;height:136px;border-radius:50%;
    background:rgba(20,28,40,.5);border:1px solid rgba(120,170,255,.22);
    touch-action:none;display:none;z-index:6}
  #knob{position:absolute;left:50%;top:50%;width:58px;height:58px;margin:-29px 0 0 -29px;
    border-radius:50%;background:rgba(140,190,255,.42);border:1px solid rgba(170,205,255,.7);
    pointer-events:none}
  #mlabel{position:fixed;left:26px;bottom:180px;font:11px/1.5 "Segoe UI",system-ui,sans-serif;
    color:#8ea6c0;display:none;z-index:6;pointer-events:none}
  @media (pointer:coarse){
    #stick,#mlabel{display:block}
    #hud{display:none!important}
    #tip{display:none!important}
  }'''
s = rep(s, '  .hidden{display:none!important}', CSS, 'stick css')

s = rep(s, '<canvas id="c"></canvas>',
        '<canvas id="c"></canvas>\n'
        '<div id="stick"><div id="knob"></div></div>\n'
        '<div id="mlabel">左摇杆移动 · 划屏幕转视角</div>', 'stick dom')

TOUCH = '''}, { passive: false });

// ---------- 触摸：左摇杆移动 + 划屏转视角 ----------
const isTouch = matchMedia('(pointer: coarse)').matches || navigator.maxTouchPoints > 0;
const stick = document.getElementById('stick'), knob = document.getElementById('knob');
const joy = { x: 0, y: 0 };
let joyId = null, lookId = null, lx = 0, ly = 0;
const SENS_T = 0.0075;
let FOVB = 1.22;
if (isTouch) { FOVB = 1.74; fovCur = 1.74; }   // 手机竖屏给更大垂直 FOV（广角）

function stickMove(t) {
  const r = stick.getBoundingClientRect(), R = r.width / 2;
  let dx = t.clientX - (r.left + R), dy = t.clientY - (r.top + R);
  const d = Math.hypot(dx, dy);
  if (d > R) { dx *= R / d; dy *= R / d; }
  knob.style.transform = `translate(${dx}px,${dy}px)`;
  joy.x = dx / R; joy.y = -dy / R;
}
stick.addEventListener('touchstart', e => {
  e.preventDefault(); e.stopPropagation();
  if (joyId !== null) return;
  const t = e.changedTouches[0];
  joyId = t.identifier; stickMove(t);
}, { passive: false });
stick.addEventListener('touchmove', e => {
  e.preventDefault(); e.stopPropagation();
  for (const t of e.changedTouches) if (t.identifier === joyId) stickMove(t);
}, { passive: false });
function endStick(e) {
  for (const t of e.changedTouches) {
    if (t.identifier === joyId) {
      joyId = null; joy.x = 0; joy.y = 0; knob.style.transform = 'translate(0,0)';
    }
  }
}
stick.addEventListener('touchend', endStick);
stick.addEventListener('touchcancel', endStick);

canvas.addEventListener('touchstart', e => {
  const t = e.changedTouches[0];
  if (lookId === null) { lookId = t.identifier; lx = t.clientX; ly = t.clientY; }
  e.preventDefault();
}, { passive: false });
canvas.addEventListener('touchmove', e => {
  for (const t of e.changedTouches) {
    if (t.identifier !== lookId) continue;
    const dx = t.clientX - lx, dy = t.clientY - ly;
    lx = t.clientX; ly = t.clientY;
    cam.yaw -= dx * SENS_T;
    cam.pitch += dy * SENS_T;
    cam.pitch = Math.max(-1.52, Math.min(1.52, cam.pitch));
  }
  e.preventDefault();
}, { passive: false });
function endLook(e) {
  for (const t of e.changedTouches) if (t.identifier === lookId) lookId = null;
}
canvas.addEventListener('touchend', endLook);
canvas.addEventListener('touchcancel', endLook);

function fwdVec() {'''
s = rep(s, '}, { passive: false });\n\nfunction fwdVec() {', TOUCH, 'touch js')

s = rep(s, "const fovT = 1.22 * (sprinting ? 1.13 : 1);",
        "const fovT = FOVB * (sprinting ? 1.13 : 1);", 'fov')

s = rep(s, "canvas.addEventListener('mousedown', e => {\n  dragging = true;",
        "canvas.addEventListener('mousedown', e => {\n  if (isTouch) return;\n  dragging = true;", 'mousedown')

s = rep(s, "canvas.addEventListener('click', () => {\n  if (!document.pointerLockElement) {",
        "canvas.addEventListener('click', () => {\n  if (isTouch) return;\n  if (!document.pointerLockElement) {", 'click')

s = rep(s, "  if (keys.ShiftLeft || keys.ShiftRight) m[1] -= 1;",
        "  if (keys.ShiftLeft || keys.ShiftRight) m[1] -= 1;\n"
        "  if (joy.y) { m[0] += Math.sin(cam.yaw) * joy.y; m[2] += -Math.cos(cam.yaw) * joy.y; }\n"
        "  if (joy.x) { m[0] += rg[0] * joy.x; m[2] += rg[2] * joy.x; }", 'move')

io.open(DST, "w", encoding="utf-8").write(s)
print("saved", DST, round(len(s) / 1e6, 2), "MB")
