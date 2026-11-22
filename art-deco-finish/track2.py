# Per-layer camera tracking (parallax-aware): features are taken only from the pixels of one layer.
import sys, json, subprocess, numpy as np, cv2
SRC, OUT = sys.argv[1], sys.argv[2]
W, H = 1920, 1080; w, h = W // 2, H // 2
dec = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-i', SRC, '-vf', f'scale={w}:{h}:flags=lanczos', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], stdout=subprocess.PIPE)
F = []
while True:
    b = dec.stdout.read(w * h * 3)
    if len(b) < w * h * 3: break
    F.append(np.frombuffer(b, np.uint8).reshape(h, w, 3))
COL = {'wall': (240, 208, 148), 'roof': (182, 78, 60), 'panel': (16, 32, 48), 'floorN': (98, 68, 66), 'table': (186, 154, 130),
       'arm': (170, 100, 80), 'candle': (110, 30, 45)}
def cmask(img, kinds, tol=26, dil=6):
    m = np.zeros(img.shape[:2], np.uint8)
    for k in kinds:
        d = np.abs(img.astype(np.int16) - np.array(COL[k], np.int16)).sum(2); m |= (d < tol).astype(np.uint8)
    if dil: m = cv2.dilate(m, np.ones((dil * 2 + 1, dil * 2 + 1), np.uint8))
    return m * 255
LAYERS = {   # name: (shot, first, last, ref, colour kinds, extra region (x0,x1) at half res or None)
    'A_house': ('A', 0, 72, 36, ['wall', 'roof'], None),
    'B_house': ('B', 72, 298, 168, ['wall', 'roof'], None),
    'C_house': ('C', 298, 410, 348, ['wall', 'roof'], None),
    'N_room': ('N', 430, 720, 456, ['panel', 'floorN'], (0, 420)),
    'N_table': ('N', 430, 720, 456, ['table'], None),
}
orb = cv2.ORB_create(2500, fastThreshold=7); bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
def est(a, b, ma, mb):
    ga, gb = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    ka, da = orb.detectAndCompute(ga, ma); kb, db = orb.detectAndCompute(gb, mb)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8: return None, 0
    m = sorted(bf.match(da, db), key=lambda x: x.distance)[:500]
    if len(m) < 8: return None, 0
    pa = np.float32([ka[x.queryIdx].pt for x in m]); pb = np.float32([kb[x.trainIdx].pt for x in m])
    M, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=1.5)
    return M, int(inl.sum()) if inl is not None else 0
res = {}
for name, (shot, s, e, ref, kinds, xr) in LAYERS.items():
    def mk(i):
        m = cmask(F[i], kinds)
        if xr: m[:, :xr[0]] = 0; m[:, xr[1]:] = 0
        return m
    T = {ref: np.eye(3)}; weak = 0
    for step in (1, -1):
        cur = np.eye(3); f = ref
        while s <= f + step < e:
            M, n = est(F[f], F[f + step], mk(f), mk(f + step)); f += step
            if M is None or n < 10: M = np.float32([[1, 0, 0], [0, 1, 0]]); weak += 1      # hold last position
            cur = np.vstack([M, [0, 0, 1]]) @ cur; T[f] = cur.copy()
    S = np.diag([2, 2, 1])
    res[name] = {str(f): (S @ M @ np.linalg.inv(S))[:2].tolist() for f, M in T.items()}
    sc = [round(float(np.sqrt(abs(np.linalg.det(T[f][:2, :2])))), 3) for f in (s, ref, e - 1)]
    tx = [round(float(T[f][0, 2] * 2)) for f in (s, ref, e - 1)]
    print(f'{name:8s} frames {s}-{e - 1} scale {sc} tx {tx} weak {weak}')
json.dump(res, open(OUT, 'w'))
