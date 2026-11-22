# Refine Codex's Art Deco cut frame by frame: 1080p, gold deco frame, night glow + candle flicker,
# gold dust, end-card sunburst, gentle drift, grain, vignette, fades. Then mux the new score.
import sys, subprocess, json, numpy as np
import details as DT
from PIL import Image, ImageDraw, ImageFilter

SRC, WAV, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
PREVIEW = [float(x) for x in sys.argv[4].split(',')] if len(sys.argv) > 4 else None
TRACK = json.load(open('track2.json'))
REFS = {k: np.asarray(Image.open(f).convert('RGB')) for k, f in (('A','ref_1.5.png'),('B','ref_7.png'),('C','ref_14.5.png'),('N','ref_19.png'))}
LAYERS = DT.build(REFS, TRACK)
print('layers built', flush=True)
W, H, FPS = 1920, 1080, 24
rng = np.random.default_rng(23)
GOLD = (214, 178, 106)

# ---- static gold Art Deco frame (double rule + stepped corner fans) ----
fr = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(fr)
for inset, wdt, a in ((34, 2, 200), (44, 1, 150)):
    d.rectangle([inset, inset, W - inset, H - inset], outline=GOLD + (a,), width=wdt)
def corner(cx, cy, sx, sy):
    for k, r in enumerate((26, 44, 62)):                                       # stepped quarter fans
        box = [cx - r, cy - r, cx + r, cy + r]
        start = {(1, 1): 0, (-1, 1): 90, (-1, -1): 180, (1, -1): 270}[(sx, sy)]
        d.arc(box, start, start + 90, fill=GOLD + (210 - 40 * k,), width=2)
    for ang in np.linspace(0, np.pi / 2, 6)[1:-1]:                              # sunray spokes
        x2 = cx + sx * 62 * np.cos(ang); y2 = cy + sy * 62 * np.sin(ang)
        d.line([cx, cy, x2, y2], fill=GOLD + (120,), width=1)
    d.rectangle([cx - 4, cy - 4, cx + 4, cy + 4], fill=GOLD + (230,))
for cx, cy, sx, sy in ((34, 34, 1, 1), (W - 34, 34, -1, 1), (34, H - 34, 1, -1), (W - 34, H - 34, -1, -1)):
    corner(cx, cy, sx, sy)
for x in (W // 2 - 60, W // 2 + 60):                                             # small chevrons top/bottom centre
    pass
d.polygon([(W // 2 - 14, 34), (W // 2, 22), (W // 2 + 14, 34)], fill=GOLD + (200,))
d.polygon([(W // 2 - 14, H - 34), (W // 2, H - 22), (W // 2 + 14, H - 34)], fill=GOLD + (200,))
FRAME = np.asarray(fr).astype(np.float32) / 255.0
FA = FRAME[..., 3:4]; FC = FRAME[..., :3]

# ---- vignette ----
yy, xx = np.mgrid[0:H, 0:W]
r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
VIG = (1 - 0.28 * np.clip(r - 0.55, 0, 1) ** 1.6)[..., None].astype(np.float32)

# ---- sunburst (end card) ----
ang = np.arctan2(yy - H * 0.47, xx - W / 2); rad = np.sqrt((xx - W / 2) ** 2 + (yy - H * 0.47) ** 2)
def sunburst(t):
    rays = (np.cos(24 * (ang + 0.035 * t)) > 0.86).astype(np.float32)
    fade = np.clip((rad - 170) / 260, 0, 1) * np.clip((1100 - rad) / 500, 0, 1)
    return (rays * fade)[..., None]

# ---- gold dust ----
P = 140
px = rng.uniform(0, W, P); py = rng.uniform(0, H, P); pv = rng.uniform(6, 22, P); ps = rng.uniform(1.2, 3.2, P); pph = rng.uniform(0, 6.3, P)
dot = {}
def dust_layer(t, amount):
    lay = Image.new('L', (W // 2, H // 2), 0); dd = ImageDraw.Draw(lay)
    for i in range(P):
        x = (px[i] + 14 * np.sin(0.4 * t + pph[i])) % W; y = (py[i] - pv[i] * t) % H
        a = int(255 * amount * (0.5 + 0.5 * np.sin(1.7 * t + pph[i])))
        s = ps[i]; dd.ellipse([x / 2 - s, y / 2 - s, x / 2 + s, y / 2 + s], fill=a)
    lay = lay.filter(ImageFilter.GaussianBlur(1.6)).resize((W, H), Image.BILINEAR)
    return (np.asarray(lay).astype(np.float32) / 255.0)[..., None]

def bloom(img, strength):
    y = img.mean(axis=2)
    m = np.clip((y - 0.78) / 0.22, 0, 1)[..., None] * img
    small = Image.fromarray((m * 255).astype(np.uint8)).resize((W // 6, H // 6), Image.BILINEAR).filter(ImageFilter.GaussianBlur(7))
    g = np.asarray(small.resize((W, H), Image.BILINEAR)).astype(np.float32) / 255.0
    return img + g * strength

def drift(img, t, seg):
    """gentle continuous zoom on each shot so holds feel alive"""
    s0, s1, z = seg
    k = (t - s0) / (s1 - s0); zoom = 1 + z * k
    if zoom <= 1.0005: return img
    pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    cw, ch = W / zoom, H / zoom; x0 = (W - cw) / 2; y0 = (H - ch) / 2
    return np.asarray(pil.resize((W, H), Image.BICUBIC, box=(x0, y0, x0 + cw, y0 + ch))).astype(np.float32) / 255.0

SEGS = [(0, 3.0, 0.035), (3.0, 12.42, 0.02), (12.42, 17.4, 0.0), (17.4, 18.0, 0.0), (18.0, 30.0, 0.0), (30.0, 35.0, 0.05)]
def seg_of(t):
    for s in SEGS:
        if s[0] <= t < s[1]: return s
    return SEGS[-1]

dec = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-i', SRC, '-vf', f'scale={W}:{H}:flags=lanczos', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], stdout=subprocess.PIPE)
enc = None if PREVIEW else subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                        '-i', WAV, '-map', '0:v', '-map', '1:a', '-c:v', 'libx264', '-preset', 'slow', '-crf', '16', '-pix_fmt', 'yuv420p',
                        '-af', 'loudnorm=I=-16:TP=-1.5:LRA=9', '-c:a', 'aac', '-b:a', '256k', '-ar', '48000', '-shortest', '-movflags', '+faststart', OUT],
                       stdin=subprocess.PIPE)
n = 0; fsz = W * H * 3
while True:
    buf = dec.stdout.read(fsz)
    if len(buf) < fsz: break
    t = n / FPS
    if PREVIEW and not any(abs(t - p) < 0.5 / FPS for p in PREVIEW): n += 1; continue
    u8 = DT.apply(np.frombuffer(buf, np.uint8).reshape(H, W, 3), n, LAYERS, TRACK)
    img = u8.astype(np.float32) / 255.0
    img = drift(img, t, seg_of(t))
    if 18.0 <= t < 30.0:                                                            # night: glow + candle flicker
        flick = 0.55 + 0.25 * np.sin(2 * np.pi * 7.3 * t) * np.sin(2 * np.pi * 1.9 * t + 1) + 0.1 * rng.standard_normal()
        img = bloom(img, max(0.25, flick if t >= 22 else 0.45))
        img = img + dust_layer(t, 0.55 * min(1, (t - 18) / 1.5)) * np.array([0.84, 0.70, 0.42], np.float32) * 0.55
    elif t >= 30.0:                                                                 # end card: sunburst + dust + glow
        a = min(1, (t - 30) / 1.2)
        img = img + sunburst(t) * np.array([0.84, 0.70, 0.42], np.float32) * 0.07 * a
        img = bloom(img, 0.25)
        img = img + dust_layer(t, 0.4 * a) * np.array([0.84, 0.70, 0.42], np.float32) * 0.5
    else:                                                                           # day: warm sun haze, very light
        img = bloom(img, 0.12)
    img = img * VIG
    fa = FA * (min(1, t / 1.0) if t < 30 else 1.0)
    img = img * (1 - fa) + FC * fa                                                  # gold frame
    img = img + rng.normal(0, 0.012, (H // 2, W // 2, 1)).repeat(2, 0).repeat(2, 1).astype(np.float32)  # film grain
    fade = min(1, t / 0.6) * min(1, (35.0 - t) / 1.2)
    if 29.6 <= t < 30.0: fade *= (30.0 - t) / 0.4                                   # dip to black into the end card
    if 30.0 <= t < 30.35: fade *= (t - 30.0) / 0.35
    img = np.clip(img * fade, 0, 1)
    if PREVIEW: Image.fromarray((img * 255).astype(np.uint8)).save(f'prev_{t:05.2f}.png')
    else: enc.stdin.write((img * 255).astype(np.uint8).tobytes())
    n += 1
    if n % 120 == 0: print('frame', n, flush=True)
if enc: enc.stdin.close(); enc.wait()
dec.wait()
print('done', n, 'frames')
