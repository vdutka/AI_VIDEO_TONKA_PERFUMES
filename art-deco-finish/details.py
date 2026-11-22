# Art Deco detail layer for «Smoky Pomegranate»: drawn in each shot's reference coordinates, warped by the
# camera track, and composited only onto the right surface (wall / sky / floor / panel) so characters stay in front.
import json, numpy as np, cv2
from PIL import Image, ImageDraw

W, H = 1920, 1080
TEAL_D = (30, 72, 82); TEAL = (52, 112, 112); GOLD = (214, 178, 106); BURG = (122, 38, 52); CREAM = (246, 226, 182)
ROSE = (200, 64, 92); ROSE_L = (232, 128, 146); LEAF = (46, 96, 84); TERRA = (178, 82, 58); TERRA_D = (140, 60, 44)
CROCUS = (128, 92, 168); CROCUS_L = (168, 140, 206); SAFFRON = (222, 92, 28); NAVY = (16, 32, 48); WOOD = (92, 52, 44)

COL = {'wall': ((240, 208, 148), 18), 'sky': ((138, 170, 162), 30), 'floorN': ((98, 68, 66), 26),
       'winsky': ((50, 66, 90), 24), 'panel': ((16, 32, 48), 22), 'table': ((186, 154, 130), 30)}
def mask(img, kinds):
    if kinds == ['not_table']: return 1 - mask(img, ['table'])
    m = np.zeros(img.shape[:2], np.float32)
    for k in kinds:
        c, tol = COL[k]; d = np.abs(img.astype(np.int16) - np.array(c, np.int16)).sum(2)
        m = np.maximum(m, np.clip((tol * 1.3 - d) / (tol * 0.3), 0, 1))
    return cv2.GaussianBlur(m, (3, 3), 0)[..., None]

class Canvas:
    def __init__(s, x0, y0, x1, y1, k):
        s.x0, s.y0, s.k = x0, y0, k
        s.im = Image.new('RGBA', (int((x1 - x0) * k), int((y1 - y0) * k)), (0, 0, 0, 0)); s.d = ImageDraw.Draw(s.im)
    def P(s, x, y): return ((x - s.x0) * s.k, (y - s.y0) * s.k)
    def box(s, x0, y0, x1, y1): return [*s.P(x0, y0), *s.P(x1, y1)]
    def rect(s, x0, y0, x1, y1, c): s.d.rectangle(s.box(x0, y0, x1, y1), fill=c)
    def ell(s, x0, y0, x1, y1, c, o=None, w=1): s.d.ellipse(s.box(x0, y0, x1, y1), fill=c, outline=o, width=max(1, int(w * s.k)))
    def poly(s, pts, c): s.d.polygon([s.P(*p) for p in pts], fill=c)
    def line(s, pts, c, w): s.d.line([s.P(*p) for p in pts], fill=c, width=max(1, int(w * s.k)))
    def arch_win(s, x, y, w, h, c=TEAL_D, sill=GOLD):
        s.rect(x, y + w / 2, x + w, y + h, c); s.d.pieslice(s.box(x, y, x + w, y + w), 180, 360, fill=c)
        s.rect(x - 2, y + h, x + w + 2, y + h + 2.5, sill)
    def array(s):
        a = np.asarray(s.im).astype(np.float32) / 255.0
        return a, np.array([[1 / s.k, 0, s.x0], [0, 1 / s.k, s.y0], [0, 0, 1]], np.float64)

# ---------------- motifs ----------------
def rose_bush(c, cx, by, h=150, n=7, seed=1):
    r = np.random.default_rng(seed)
    c.poly([(cx - 46, by - 78), (cx + 46, by - 78), (cx + 34, by), (cx - 34, by)], TERRA)          # pot
    c.rect(cx - 52, by - 90, cx + 52, by - 76, TERRA_D)
    for k in range(3): c.line([(cx - 36 + 4 * k, by - 50 + 14 * k), (cx + 36 - 4 * k, by - 50 + 14 * k)], GOLD, 2)
    for k in range(9):                                                                              # leaves fan
        a = np.pi * (0.15 + 0.7 * k / 8); L = h * 0.55
        tip = (cx + L * np.cos(a) * 1.1 * (1 if k % 2 else -1) * 0.6, by - 90 - L * np.sin(a))
        c.poly([(cx, by - 88), (tip[0] - 9, tip[1] + 18), tip, (tip[0] + 9, tip[1] + 18)], LEAF)
    for k in range(n):                                                                              # deco roses: concentric
        x = cx + r.uniform(-48, 48); y = by - 100 - r.uniform(10, h * 0.7); R = r.uniform(11, 16)
        c.ell(x - R, y - R, x + R, y + R, ROSE)
        c.ell(x - R * .62, y - R * .62, x + R * .62, y + R * .62, ROSE_L)
        c.d.arc(c.box(x - R * .62, y - R * .62, x + R * .62, y + R * .62), 200, 340, fill=ROSE, width=max(1, int(2 * c.k)))
        c.ell(x - R * .22, y - R * .22, x + R * .22, y + R * .22, ROSE)

def crocus_pot(c, cx, by, seed=2):
    r = np.random.default_rng(seed)
    c.poly([(cx - 34, by - 58), (cx + 34, by - 58), (cx + 26, by), (cx - 26, by)], TERRA)
    c.rect(cx - 38, by - 66, cx + 38, by - 56, TERRA_D); c.line([(cx - 28, by - 30), (cx + 28, by - 30)], GOLD, 2)
    for k in range(6):
        x = cx - 26 + k * 10.5; top = by - 110 - r.uniform(0, 30)
        c.line([(x, by - 66), (x + r.uniform(-6, 6), top + 20)], LEAF, 3)
        fx = x + r.uniform(-6, 6)
        c.poly([(fx - 11, top + 4), (fx, top - 18), (fx + 11, top + 4), (fx, top + 22)], CROCUS)      # petals
        c.poly([(fx - 5, top + 6), (fx, top - 10), (fx + 5, top + 6), (fx, top + 16)], CROCUS_L)
        for s in (-4, 0, 4): c.line([(fx, top + 6), (fx + s, top - 14)], SAFFRON, 2)                  # saffron stigmas

def skyline(c, img, xs, kinds=None):
    sky = mask(img, ['sky'])[..., 0] > 0.5
    roofs = (np.abs(img.astype(np.int16) - np.array((182, 78, 60), np.int16)).sum(2) < 40)
    cols = [np.where(roofs[:, x])[0] for x in range(0, W, 8)]
    tops = [cc.min() for cc in cols if len(cc)]
    if not tops: return
    hz = int(np.percentile(tops, 30))                                              # roofline height
    tone = (124, 156, 150); far = (114, 146, 142)
    band_top = hz - 34
    c.rect(0, band_top, W, hz + 160, tone)                                         # distant city wall (hidden below by houses)
    for x in range(0, W, 26): c.rect(x, band_top - 9, x + 13, band_top, tone)      # crenellations
    for x in range(40, W, 210): c.rect(x, band_top - 46, x + 34, band_top, tone); c.poly([(x - 4, band_top - 46), (x + 17, band_top - 62), (x + 38, band_top - 46)], tone)
    dx = xs[0]; R = 50                                                              # dome on a drum, standing on the wall
    c.rect(dx - R * 0.8, band_top - R * 0.9, dx + R * 0.8, band_top, far)
    c.d.pieslice(c.box(dx - R, band_top - R * 2.0, dx + R, band_top - R * 0.1), 180, 360, fill=far)
    c.line([(dx, band_top - R * 2.0), (dx, band_top - R * 2.4)], far, 3)
    for k in range(3): c.arch_win(dx - R * 0.55 + k * R * 0.45, band_top - R * 0.7, 9, 18, (100, 132, 128), (100, 132, 128))
    mx = xs[1]                                                                      # minaret standing on the wall
    c.rect(mx - 11, band_top - 200, mx + 11, band_top, far); c.rect(mx - 17, band_top - 160, mx + 17, band_top - 150, far)
    c.poly([(mx - 11, band_top - 200), (mx, band_top - 236), (mx + 11, band_top - 200)], far)

def house_details(c, img, y0, y1, seed=3):
    wall = (mask(img, ['wall'])[..., 0] > 0.6).astype(np.uint8); wall[:y0] = 0; wall[y1:] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(wall, 8)
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if w < 46 or h < 30 or area < 1500: continue
        c.rect(x, y, x + w, y + max(3, h * 0.045), TEAL_D)                       # cornice under the roof
        ww = float(np.clip(w * 0.12, 9, 20)); hh = ww * 1.7; wy = y + h * 0.28
        for fx in (0.27, 0.73):                                                   # symmetric pair of arched windows
            wx = x + w * fx - ww / 2
            patch = wall[int(wy) - 2:int(wy + hh) + 4, int(wx) - 3:int(wx + ww) + 3]
            if patch.size and patch.mean() > 0.97: c.arch_win(wx, wy, ww, hh)

def chimneys(c, img, y0, y1, seed=12):
    d = np.abs(img.astype(np.int16) - np.array((182, 78, 60), np.int16)).sum(2)
    roof = ((d < 40).astype(np.uint8)); roof[:y0] = 0; roof[y1:] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(roof, 8)
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if w < 50 or area < 600: continue
        cx = x + w * 0.72; top = y + h * 0.42; cw = max(7, w * 0.075); ch = max(16, h * 0.85)
        c.rect(cx - cw / 2, top - ch, cx + cw / 2, top + 8, (226, 196, 140))
        c.rect(cx - cw / 2 - 3, top - ch - 5, cx + cw / 2 + 3, top - ch + 1, (150, 70, 52))

def bookshelf(c, x0, y0, x1, y1, seed=4):
    r = np.random.default_rng(seed)
    c.rect(x0, y0, x1, y1, WOOD); rows = 3; rh = (y1 - y0) / rows
    for k in range(rows):
        top = y0 + k * rh + 6; bot = y0 + (k + 1) * rh - 4; x = x0 + 6
        while x < x1 - 12:
            bw = r.uniform(7, 13); bh = r.uniform(0.6, 1.0) * (bot - top)
            colr = [BURG, GOLD, TEAL, CREAM, (70, 50, 90), ROSE][r.integers(0, 6)]
            c.rect(x, bot - bh, x + bw, bot, colr); c.rect(x, bot - bh + 6, x + bw, bot - bh + 8, GOLD if colr != GOLD else BURG)
            x += bw + 1.5
        c.rect(x0, y0 + (k + 1) * rh - 4, x1, y0 + (k + 1) * rh + 1, (60, 34, 30))
    c.poly([(x0 - 6, y0), (x1 + 6, y0), (x1, y0 + 8), (x0, y0 + 8)], GOLD)

def gramophone(c, cx, by):
    c.rect(cx - 6, by - 90, cx + 6, by, WOOD); c.ell(cx - 50, by - 104, cx + 50, by - 86, WOOD)        # side table
    c.rect(cx - 42, by - 132, cx + 42, by - 100, BURG); c.line([(cx - 42, by - 116), (cx + 42, by - 116)], GOLD, 2)
    c.ell(cx - 36, by - 140, cx + 36, by - 128, NAVY)                                                  # record
    c.line([(cx - 20, by - 134), (cx - 6, by - 168)], GOLD, 4)                                         # arm
    c.poly([(cx - 6, by - 168), (cx + 52, by - 214), (cx + 70, by - 176), (cx, by - 160)], GOLD)       # horn toward the armchair
    c.poly([(cx - 2, by - 166), (cx + 50, by - 204), (cx + 62, by - 180), (cx + 2, by - 162)], (236, 204, 138))

def rug(c, cx, cy, rx, ry):
    c.ell(cx - rx, cy - ry, cx + rx, cy + ry, BURG); c.ell(cx - rx + 10, cy - ry + 5, cx + rx - 10, cy + ry - 5, None, GOLD, 2)
    c.ell(cx - rx * .55, cy - ry * .55, cx + rx * .55, cy + ry * .55, None, GOLD, 1.5)
    for k in range(12):
        a = 2 * np.pi * k / 12; c.line([(cx + rx * .58 * np.cos(a), cy + ry * .58 * np.sin(a)), (cx + rx * .85 * np.cos(a), cy + ry * .85 * np.sin(a))], GOLD, 1.2)

def vase_roses(c, cx, by, h=190):
    c.poly([(cx - 16, by), (cx + 16, by), (cx + 24, by - 60), (cx + 10, by - 92), (cx - 10, by - 92), (cx - 24, by - 60)], TEAL)
    for k in range(3): c.line([(cx - 20 + 2 * k, by - 30 - 14 * k), (cx + 20 - 2 * k, by - 30 - 14 * k)], GOLD, 1.5)
    r = np.random.default_rng(7)
    for k, (dx, dy) in enumerate(((-30, -150), (0, -180), (28, -146), (-12, -122), (16, -112))):
        c.line([(cx, by - 90), (cx + dx, by + dy)], LEAF, 2.5)
        c.poly([(cx + dx * .5, by - 110 + dy * .2), (cx + dx * .5 - 14, by - 104 + dy * .2), (cx + dx * .5, by - 96 + dy * .2)], LEAF)
        R = 13 - k * .8; x, y = cx + dx, by + dy
        c.ell(x - R, y - R, x + R, y + R, ROSE); c.ell(x - R * .6, y - R * .6, x + R * .6, y + R * .6, ROSE_L)
        c.ell(x - R * .22, y - R * .22, x + R * .22, y + R * .22, ROSE)

def palm(c, cx, by):
    c.poly([(cx - 30, by - 70), (cx + 30, by - 70), (cx + 22, by), (cx - 22, by)], GOLD)
    c.line([(cx - 24, by - 40), (cx + 24, by - 40)], NAVY, 3)
    for k in range(9):
        a = np.pi * (0.1 + 0.8 * k / 8); L = 150 + 40 * np.sin(np.pi * k / 8)
        tip = (cx + L * np.cos(a), by - 70 - L * np.sin(a))
        mid = (cx + L * .5 * np.cos(a) - 12, by - 70 - L * .5 * np.sin(a) - 10)
        c.poly([(cx, by - 70), mid, tip, (mid[0] + 22, mid[1] + 14)], LEAF)

def sconce(c, cx, cy):
    c.poly([(cx - 18, cy), (cx + 18, cy), (cx + 8, cy + 30), (cx - 8, cy + 30)], GOLD)
    for k in range(5): c.line([(cx, cy), (cx - 26 + 13 * k, cy - 26)], GOLD, 1.5)

def petals_on_table(c, pts, seed=9):
    r = np.random.default_rng(seed)
    for (x, y) in pts:
        for k in range(3):
            px, py = x + r.uniform(-14, 14), y + r.uniform(-6, 6); s = r.uniform(5, 8)
            c.ell(px - s, py - s * .55, px + s, py + s * .55, ROSE if k % 2 else ROSE_L)
        for k in range(5):
            px, py = x + r.uniform(-18, 18), y + r.uniform(-5, 5); a = r.uniform(0, np.pi)
            c.line([(px, py), (px + 7 * np.cos(a), py + 3 * np.sin(a))], SAFFRON, 1.2)

def saffron_dish(c, cx, cy):
    c.ell(cx - 26, cy - 8, cx + 26, cy + 8, GOLD); c.ell(cx - 20, cy - 6, cx + 20, cy + 4, (196, 150, 80))
    r = np.random.default_rng(11)
    for k in range(18):
        x, y = cx + r.uniform(-15, 15), cy + r.uniform(-4, 2); a = r.uniform(0, np.pi)
        c.line([(x, y), (x + 6 * np.cos(a), y + 2.5 * np.sin(a))], SAFFRON, 1.1)

# ---------------- build per-shot layers ----------------
def build(refs, track):
    L = []   # (layer_track_name, canvas, mask kinds)
    A = refs['A']
    ca = Canvas(0, 0, W, H, 1); house_details(ca, A, 270, 450, 3); L.append(('A_house', ca, ['wall']))
    cc_ = Canvas(0, 0, W, H, 1); chimneys(cc_, A, 260, 450); L.append(('A_house', cc_, ['sky']))
    cs = Canvas(0, 0, W, H, 1); skyline(cs, A, [1180, 1400, 1700]); L.append(('A_house', cs, ['sky']))
    cp = Canvas(1450, 700, 1920, 1060, 2); rose_bush(cp, 1630, 1010); crocus_pot(cp, 1830, 1020); L.append(('A_house', cp, None))
    ct = Canvas(600, 800, 1500, 1000, 2); petals_on_table(ct, [(650, 905), (1080, 915), (1330, 880)]); L.append(('A_house', ct, None))
    B = refs['B']
    cb = Canvas(0, 0, W, H, 1); house_details(cb, B, 100, 330, 5); L.append(('B_house', cb, ['wall']))
    cbc = Canvas(0, 0, W, H, 1); chimneys(cbc, B, 60, 330); L.append(('B_house', cbc, ['sky']))
    cbs = Canvas(0, 0, W, H, 1); skyline(cbs, B, [1290, 1600, 1830]); L.append(('B_house', cbs, ['sky']))
    C = refs['C']
    cc = Canvas(0, 0, W, H, 3); house_details(cc, C, 60, 420, 6); L.append(('C_house', cc, ['wall']))
    ccc = Canvas(0, 0, W, H, 3); chimneys(ccc, C, 40, 420); L.append(('C_house', ccc, ['sky']))
    ccs = Canvas(0, 0, W, H, 3); skyline(ccs, C, [520, 980, 1500]); L.append(('C_house', ccs, ['sky']))
    N = refs['N']
    tab = mask(N, ['table'])[..., 0] > 0.5; ys = np.where(tab[:, 980])[0]
    vy = int(np.percentile(ys, 40)) if len(ys) else 760
    cn2 = Canvas(0, 150, 1100, 1000, 2); rug(cn2, 330, 770, 240, 46); L.append(('N_room', cn2, ['floorN']))
    cn1 = Canvas(0, 150, 1100, 1000, 2); bookshelf(cn1, 24, 450, 186, 610); sconce(cn1, 105, 435); gramophone(cn1, 100, 815)
    L.append(('N_room', cn1, None))
    cn3 = Canvas(0, 150, 1100, 1000, 2); palm(cn3, 860, 720); L.append(('N_room', cn3, ['not_table']))
    cn5 = Canvas(700, 350, 1300, 1000, 3); vase_roses(cn5, 980, vy); L.append(('N_table', cn5, None))
    Mi = np.linalg.inv(np.array(track['N_table']['624'] + [[0, 0, 1]]))
    back = lambda x, y: tuple((Mi @ np.array([x, y, 1]))[:2])
    cn4 = Canvas(1100, 500, 1920, 900, 4)
    sx, sy = back(800, 700); saffron_dish(cn4, sx, sy)
    petals_on_table(cn4, [back(1560, 1010), back(1760, 1040), back(150, 1000), back(700, 1050), back(1200, 1060)])
    L.append(('N_table', cn4, None))
    return [(name, c.array(), kinds) for name, c, kinds in L]

def apply(img_u8, frame_no, layers, track):
    img = None
    for name, (rgba, C2R), kinds in layers:
        Mt = track.get(name, {}).get(str(frame_no))
        if Mt is None: continue
        if img is None: img = img_u8.astype(np.float32) / 255.0
        A2 = (np.array(Mt + [[0, 0, 1]]) @ C2R)[:2]
        warped = cv2.warpAffine(rgba, A2, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        a = warped[..., 3:4]
        if a.max() <= 0.001: continue
        if name == 'C_house':
            sc = float(np.sqrt(abs(np.linalg.det(np.array(Mt)[:, :2]))))
            a = a * float(np.clip((1.25 - sc) / 0.2, 0, 1))
        if kinds: a = a * mask(img_u8, kinds)
        img = img * (1 - a) + warped[..., :3] * a
    return img_u8 if img is None else (np.clip(img, 0, 1) * 255).astype(np.uint8)

# ---------------- screen-space scent particles: rose petals + saffron threads ----------------
class Scent:
    def __init__(s, seed=17):
        r = np.random.default_rng(seed); s.p = []
        for k in range(70):                                         # hand-over / she smells (B)
            s.p.append(dict(shot='B', ref=(955, 650), t0=6.6 + r.uniform(0, 5.0), life=r.uniform(2.2, 3.4), vx=r.uniform(-10, 60), vy=r.uniform(-130, -70),
                            sw=r.uniform(15, 40), ph=r.uniform(0, 6.3), kind='petal' if r.random() < 0.55 else 'thread', rot=r.uniform(0, 6.3), spin=r.uniform(-2, 2), sz=r.uniform(6, 11)))
        for k in range(60):                                         # night: rising from pomegranate + candle (N)
            s.p.append(dict(shot='N', ref=(1270 + r.uniform(-60, 120), 720), t0=22.5 + r.uniform(0, 6.5), life=r.uniform(2.5, 3.8), vx=r.uniform(-20, 30), vy=r.uniform(-90, -50),
                            sw=r.uniform(10, 30), ph=r.uniform(0, 6.3), kind='petal' if r.random() < 0.5 else 'thread', rot=r.uniform(0, 6.3), spin=r.uniform(-1.6, 1.6), sz=r.uniform(7, 12)))
    def layer(s, t, frame_no, track):
        tr = track.get(str(frame_no))
        if not tr: return None
        M = np.array(tr['M'] + [[0, 0, 1]]); sc = np.sqrt(abs(np.linalg.det(M[:2, :2])))
        im = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(im); any_ = False
        for q in s.p:
            if q['shot'] != tr['shot']: continue
            age = t - q['t0']
            if age < 0 or age > q['life']: continue
            ox, oy = (M @ np.array([q['ref'][0], q['ref'][1], 1]))[:2]
            x = ox + q['vx'] * age + q['sw'] * np.sin(1.8 * age + q['ph']); y = oy + q['vy'] * age
            a = int(235 * min(1, age / 0.5) * min(1, (q['life'] - age) / 0.8)); rot = q['rot'] + q['spin'] * age
            sz = q['sz'] * min(2.2, 0.7 + 0.3 * sc)
            if q['kind'] == 'petal':
                us = np.linspace(0, 2 * np.pi, 18, endpoint=False)
                ex = sz * np.cos(us) * (1 + 0.35 * np.cos(us)); ey = 0.55 * sz * np.sin(us) * (0.75 + 0.25 * np.cos(us))   # teardrop petal
                flip = 0.35 + 0.65 * abs(np.sin(1.3 * age + q['ph']))                                                 # tumbling
                pts = [(x + a1 * np.cos(rot) - b1 * flip * np.sin(rot), y + a1 * np.sin(rot) + b1 * flip * np.cos(rot)) for a1, b1 in zip(ex, ey)]
                d.polygon(pts, fill=ROSE_L + (a,) if int(q['ph'] * 10) % 2 else ROSE + (a,))
            else:
                L = sz * 1.6; d.line([(x, y), (x + L * np.cos(rot), y + L * np.sin(rot))], fill=SAFFRON + (a,), width=max(2, int(sz / 4)))
            any_ = True
        if not any_: return None
        return np.asarray(im).astype(np.float32) / 255.0
