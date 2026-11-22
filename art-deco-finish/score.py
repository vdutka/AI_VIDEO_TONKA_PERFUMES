# «Smoky Pomegranate» Art Deco cut: original 1920s Baku jazz score + sound design, synthesised from scratch.
# Picture map: 0-3 courtyard wide | 3-12.4 hand-over, she smells it | 12.4-17.5 cat | ~17.5 black (fur) |
#              18-30 night flat, candle push-in | 30-35 end card.
import sys, numpy as np
from scipy.signal import oaconvolve, butter, sosfilt
from scipy.io import wavfile

SR = 48000; DUR = 35.0; N = int(DUR * SR)
rng = np.random.default_rng(1920)
L = np.zeros(N); R = np.zeros(N)
hz = lambda m: 440.0 * 2 ** ((m - 69) / 12)
def T(d): return np.arange(int(d * SR)) / SR
def bp(x, lo, hi, o=2): return sosfilt(butter(o, [lo, hi], 'bandpass', fs=SR, output='sos'), x)
def lp(x, f, o=2): return sosfilt(butter(o, f, 'lowpass', fs=SR, output='sos'), x)
def hp(x, f, o=2): return sosfilt(butter(o, f, 'highpass', fs=SR, output='sos'), x)
def add(sig, t, pan=0.0, g=1.0):
    i = int(t * SR)
    if i < 0: sig = sig[-i:]; i = 0
    j = min(i + len(sig), N)
    if j <= i: return
    s = sig[:j - i] * g
    L[i:j] += s * np.sqrt((1 - pan) / 2); R[i:j] += s * np.sqrt((1 + pan) / 2)
def env_ad(n, a, d):
    t = np.arange(n) / SR; e = np.exp(-t / d); k = int(a * SR)
    if k: e[:k] *= np.linspace(0, 1, k)
    return e

# ---------- time-varying music gain (automation) ----------
def curve(points):
    t = np.arange(N) / SR; xs, ys = zip(*points); return np.interp(t, xs, ys)
band = curve([(0, 0), (2.6, 0), (3.0, 1), (12.2, 1), (12.6, 0.0), (35, 0)])          # full swing band
cat = curve([(0, 0), (12.3, 0), (12.6, 1), (16.9, 1), (17.6, 0), (35, 0)])           # tiptoe section
night = curve([(0, 0), (17.8, 0), (18.4, 1), (29.6, 1), (30.2, 0.0), (35, 0)])       # intimate night
MUS = {k: (np.zeros(N), np.zeros(N)) for k in ('band', 'cat', 'night', 'free')}
def madd(bus, sig, t, pan=0.0, g=1.0):
    global L, R
    sL, sR = L, R; L, R = MUS[bus]; add(sig, t, pan, g); L, R = sL, sR

# ---------- instruments ----------
def piano(m, d=1.6, vel=1.0):
    t = T(d); f = hz(m); s = np.zeros(len(t))
    for k in range(1, 9):
        fk = f * k * np.sqrt(1 + 0.0004 * k * k)
        s += (0.9 ** k) / k * np.sin(2 * np.pi * fk * t) * np.exp(-t * (1.3 + 0.9 * k) * (1 + m / 90))
    s += bp(rng.normal(0, 1, len(t)), 800, 5000) * np.exp(-t * 90) * 0.08
    return s * vel * env_ad(len(t), 0.002, 9)

def bass(m, d=0.6, vel=1.0):
    t = T(d); f = hz(m)
    s = np.sin(2 * np.pi * f * t) + 0.45 * np.sin(4 * np.pi * f * t) * np.exp(-t * 6) + 0.2 * np.sin(6 * np.pi * f * t) * np.exp(-t * 10)
    s += bp(rng.normal(0, 1, len(t)), 200, 1500) * np.exp(-t * 60) * 0.15            # finger pluck
    return s * np.exp(-t * 3.2) * vel * env_ad(len(t), 0.004, 50)

def clarinet(notes, t0, pan=0.15, g=1.0, mute=False):
    """notes: list of (start_beat_offset_s, dur_s, midi). Legato line with portamento."""
    total = max(s + d for s, d, _ in notes) + 0.4; t = T(total); n = len(t)
    f = np.zeros(n); amp = np.zeros(n)
    for i, (s, d, m) in enumerate(notes):
        a, b = int(s * SR), int((s + d) * SR)
        f[a:] = hz(m)
        if i and abs(m - notes[i - 1][2]) <= 3:                                         # short glide (mugham feel)
            gl = int(0.06 * SR); f[a:a + gl] = np.linspace(hz(notes[i - 1][2]), hz(m), gl)
        e = np.ones(b - a); k = int(0.05 * SR); e[:k] = np.linspace(0.2, 1, k); r = int(0.08 * SR); e[-r:] *= np.linspace(1, 0.35, r)
        amp[a:b] = np.maximum(amp[a:b], e)
    amp = lp(amp, 18)
    vib = 1 + 0.006 * np.sin(2 * np.pi * 5.4 * t) * np.clip(t / 0.6, 0, 1)
    ph = 2 * np.pi * np.cumsum(f * vib) / SR
    s = np.sin(ph) + 0.42 * np.sin(3 * ph) + 0.22 * np.sin(5 * ph) + 0.1 * np.sin(7 * ph) + 0.05 * np.sin(2 * ph)
    s += bp(rng.normal(0, 1, n), 1500, 6000) * 0.05                                      # breath
    s = lp(s * amp, 2600 if mute else 4200)
    if mute: s = bp(s, 300, 2400)
    return s * g

def brush_swish(d=0.45):
    t = T(d); n = bp(rng.normal(0, 1, len(t)), 2500, 9000)
    return n * np.sin(np.pi * np.clip(t / d, 0, 1)) ** 1.5 * 0.35
def brush_tap():
    t = T(0.12); return bp(rng.normal(0, 1, len(t)), 1800, 7000) * np.exp(-t * 45) * 0.5
def ride():
    t = T(0.9); s = np.zeros(len(t))
    for fr, a in ((3170, .5), (4325, .35), (5940, .3), (7380, .2), (8710, .15)):
        s += a * np.sin(2 * np.pi * fr * t + rng.uniform(0, 6))
    s += hp(rng.normal(0, 1, len(t)), 6000) * 0.3
    return s * np.exp(-t * 4.5) * env_ad(len(t), 0.001, 9) * 0.25
def kick_soft():
    t = T(0.25); f = 55 + 40 * np.exp(-t * 30)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 14) * 0.6

# ---------- harmony: D minor, 1920s swing, 104 bpm ----------
BPM = 104; B = 60 / BPM; SW = 0.62                                                       # swing ratio
CH = [('Dm6', [50, 53, 57, 59], [38, 41, 45, 47]), ('Gm7', [53, 55, 58, 62], [43, 46, 50, 53]),
      ('A7b9', [52, 55, 58, 61], [45, 49, 52, 55]), ('Dm6', [50, 53, 57, 59], [38, 41, 45, 47])]

# 0.0-3.0 piano pickup: arpeggio rising, rubato
for i, m in enumerate([62, 65, 69, 71, 74, 73, 69]):
    madd('free', piano(m, 2.0, 0.42 - 0.02 * i), 0.55 + i * 0.27, pan=-0.2)
madd('free', piano(61, 2.5, 0.35), 2.45, -0.2); madd('free', piano(45, 2.5, 0.45), 2.45, -0.1)

# 3.0-12.4 swing band (2 bars per chord)
t0 = 3.0; bar = 4 * B
for c in range(5):
    name, voic, walk = CH[c % 4]; tb = t0 + c * bar
    for beat in range(4):                                                                 # walking bass
        madd('band', bass(walk[beat] if beat < 3 else walk[3] + (1 if c % 2 else -1), B * 0.95, 0.9), tb + beat * B, pan=0.05)
    for beat, off in ((0, 0), (1, SW * B)):                                               # charleston comp
        for m in voic: madd('band', piano(m + 12, 0.7, 0.16), tb + beat * B + off, pan=-0.3)
    for m in voic: madd('band', piano(m + 12, 0.5, 0.12), tb + 2 * B + SW * B, pan=-0.3)
    for beat in range(4):                                                                 # ride: ding, ding-a
        madd('band', ride(), tb + beat * B, pan=0.35, g=0.8 if beat % 2 else 0.6)
        if beat % 2: madd('band', ride(), tb + beat * B + SW * B, pan=0.35, g=0.45)
        if beat % 2: madd('band', brush_swish(), tb + beat * B - 0.05, pan=-0.1)
        else: madd('band', kick_soft(), tb + beat * B, g=0.5)
        madd('band', brush_tap(), tb + beat * B + SW * B, pan=-0.15, g=0.35)
# clarinet melody: D harmonic minor with augmented second (Bb-C#), Shur-like turns
q = B
mel = [(0, q * .6, 74), (q * .62, q * .38, 76), (q, q * 1.5, 77), (q * 2.5, q * .5, 76), (q * 3, q, 74),
       (q * 4, q * .6, 73), (q * 4.62, q * .38, 74), (q * 5, q * 2, 70), (q * 7, q, 69),
       (q * 8, q * .6, 69), (q * 8.62, q * .38, 70), (q * 9, q * .62, 73), (q * 9.62, q * .38, 74), (q * 10, q * 2, 76),
       (q * 12, q * .6, 77), (q * 12.62, q * .38, 76), (q * 13, q * .6, 74), (q * 13.62, q * .38, 73), (q * 14, q * 2, 74)]
madd('band', clarinet(mel, 0), t0 + bar * 0.5, pan=0.2, g=0.32)

# 12.4-17.5 cat: tiptoe pizzicato in octaves + celesta-like piano
for i, m in enumerate([50, 57, 53, 57, 52, 57, 53, 49, 50, 57]):
    madd('cat', bass(m + 12, 0.35, 0.7), 12.5 + i * B * 0.5, pan=0.1)
    if i % 2 == 0: madd('cat', piano(m + 36, 0.9, 0.12), 12.5 + i * B * 0.5 + 0.02, pan=-0.4)
for i in range(10): madd('cat', brush_tap(), 12.5 + i * B * 0.5 + 0.18, pan=0.3, g=0.25)

# 18-30 night: slow (piano rubato + muted clarinet), A major-ish turn to D minor warmth
for t_, ch in ((18.3, [38, 50, 53, 57, 60, 64]), (21.3, [43, 50, 53, 58, 62]), (24.3, [46, 53, 57, 62, 65]), (27.0, [45, 49, 55, 57, 61, 64])):
    for k, m in enumerate(ch): madd('night', piano(m, 3.4, 0.22 if m > 45 else 0.32), t_ + k * 0.045, pan=-0.25 + 0.1 * k)
nmel = [(0, 0.9, 69), (0.95, 0.45, 70), (1.45, 1.6, 69), (3.3, 0.6, 65), (3.95, 0.6, 67), (4.6, 1.7, 69),
        (6.6, 0.5, 70), (7.15, 0.5, 73), (7.7, 2.2, 74)]
madd('night', clarinet(nmel, 0, mute=True), 19.4, pan=0.25, g=0.28)
for i in range(12): madd('night', bass([38, 38, 43, 43, 46, 46, 45, 45, 38, 38, 45, 45][i], 1.0, 0.45), 18.3 + i * 1.0)

# 30-35 end card: final Dm(add9) chord + clarinet sigh
for k, m in enumerate([38, 50, 57, 62, 64, 65, 69]): madd('free', piano(m, 5.0, 0.3 if m < 60 else 0.2), 30.05 + k * 0.03, pan=-0.2 + 0.07 * k)
madd('free', clarinet([(0, 0.5, 76), (0.5, 2.6, 74)], 0, mute=True), 30.6, pan=0.25, g=0.22)

# ---------- sound design ----------
def babble(d, g=1.0):
    n = int(d * SR); out = np.zeros(n); t = np.arange(n) / SR
    for v in range(9):                                                                    # 9 distant voices
        f0 = rng.uniform(110, 240); syl = rng.uniform(3.5, 5.5)
        am = np.clip(np.sin(2 * np.pi * syl * t + rng.uniform(0, 6)) + 0.3 * np.sin(2 * np.pi * 0.7 * t + v), 0, None) ** 1.5
        am *= np.clip(np.sin(2 * np.pi * rng.uniform(0.08, 0.2) * t + v) + 0.4, 0, None)
        buzz = np.sign(np.sin(2 * np.pi * f0 * (1 + 0.04 * np.sin(2 * np.pi * 1.3 * t + v)) * t))
        voice = bp(buzz, 300, 900) * 0.6 + bp(buzz, 1100, 2600) * 0.3
        out += lp(voice * am, 2200) * (1 if v % 2 else 0.6)
    return out / 9 * g
def bird(t0, pan):
    t = T(0.18); f = 3200 + 1800 * np.sin(np.pi * t / 0.18) + 400 * np.sin(2 * np.pi * 38 * t)
    add(np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(np.pi * t / 0.18) ** 2 * 0.07, t0, pan)
def hoof(t0, pan, g):
    t = T(0.08); s = bp(rng.normal(0, 1, len(t)), 500, 2500) * np.exp(-t * 70) + np.sin(2 * np.pi * 300 * t) * np.exp(-t * 60) * 0.4
    add(lp(s, 1800) * g, t0, pan)
def glass(t0, pan, g, base=2400):
    t = T(1.6); s = np.zeros(len(t))
    for r, a, d in ((1, 1, 1.5), (2.32, .6, 3), (4.1, .35, 5), (5.9, .2, 7)):
        s += a * np.sin(2 * np.pi * base * r * t + rng.uniform(0, 6)) * np.exp(-t * d)
    add(s * 0.08 * g, t0, pan)
def crack(t0):
    t = T(0.5); s = np.zeros(len(t))
    for k in range(14):                                                                   # fibrous tearing
        i = int(rng.uniform(0, 0.28) * SR); seg = bp(rng.normal(0, 1, 600), 1200, 6000) * np.exp(-np.arange(600) / 90)
        s[i:i + 600] += seg * rng.uniform(0.3, 1)
    s += lp(rng.normal(0, 1, len(t)), 500) * np.exp(-t * 18) * 0.6                       # juicy body
    s += bp(rng.normal(0, 1, len(t)), 300, 1500) * np.exp(-((t - 0.3) / 0.08) ** 2) * 0.3  # wet squelch
    add(s * 0.35, t0, -0.05)
def inhale(t0, d=1.4):
    t = T(d); s = bp(rng.normal(0, 1, len(t)), 700, 4000) * np.sin(np.pi * t / d) ** 2
    add(s * 0.05, t0, 0.1)
def whoosh(t0, d=1.3):
    t = T(d); n = rng.normal(0, 1, len(t)); out = np.zeros(len(t)); seg = int(0.02 * SR)
    for i in range(0, len(t) - seg, seg):
        x = i / len(t); fc = 300 + 3200 * np.sin(np.pi * x) ** 2
        out[i:i + seg] = bp(n[i:i + seg + 200], fc * 0.6, fc * 1.4)[:seg]
    add(out * np.sin(np.pi * t / d) ** 1.2 * 0.22, t0, 0.0)
    t2 = T(0.9); add(bp(rng.normal(0, 1, len(t2)), 200, 1200) * np.sin(np.pi * t2 / 0.9) * 0.08, t0 + 0.2)  # fur rustle
def paw(t0, pan):
    t = T(0.07); add(lp(rng.normal(0, 1, len(t)), 900) * np.exp(-t * 60) * 0.06, t0, pan)

# courtyard 0-12.4: babble, birds, distant horse cart, tea glass clink, a far street bell
amb = babble(13.2, 0.10); amb *= np.clip(np.linspace(0, 1, len(amb)) * 6, 0, 1); amb[-int(0.8 * SR):] *= np.linspace(1, 0, int(0.8 * SR))
add(amb, 0.0, -0.2, 1.0); add(babble(13.2, 0.07), 0.0, 0.35)
for t_ in (0.9, 1.15, 4.6, 4.8, 8.3, 10.9, 11.1): bird(t_, rng.uniform(-0.6, 0.6))
for k in range(26):
    tt = 1.6 + k * 0.38 + (0.1 if k % 2 else 0); hoof(tt, 0.6 - 0.04 * k, 0.05 * np.sin(np.pi * k / 26) + 0.01)
glass(1.0, -0.3, 1.0); glass(1.12, -0.3, 0.5, 2900)
crack(5.2); inhale(7.4)
# cat 12.4-17.5: paws, courtyard falls away
for k in range(9): paw(12.7 + k * 0.5, 0.2)
whoosh(16.9)
# night 18-30: low city hum, distant traffic swells, a tram bell far away, candle crackle near the end
tn = T(12.5); hum = lp(rng.normal(0, 1, len(tn)), 180) * 0.25 + 0.04 * np.sin(2 * np.pi * 50 * tn)
hum *= np.clip(tn / 1.2, 0, 1) * np.clip((12.5 - tn) / 0.8, 0, 1); add(hum, 17.9, 0, 0.6)
for t_, p in ((20.2, -0.7), (25.1, 0.6)):
    td = T(3.5); add(bp(rng.normal(0, 1, len(td)), 150, 900) * np.sin(np.pi * td / 3.5) ** 2 * 0.07, t_, p)
glass(21.6, 0.7, 0.35, 1800); glass(21.85, 0.7, 0.3, 1800)
tc = T(8.0); cr = np.zeros(len(tc))
for k in range(60):
    i = int(rng.uniform(0, 7.8) * SR); cr[i:i + 300] += bp(rng.normal(0, 1, 300), 2000, 7000) * np.exp(-np.arange(300) / 40) * rng.uniform(0.1, 1)
cr += lp(rng.normal(0, 1, len(tc)), 300) * 0.15 * (1 + 0.5 * np.sin(2 * np.pi * 3.1 * tc))
add(cr * np.clip(tc / 1.0, 0, 1) * 0.05, 22.0, 0.1)
# end card: spoon on tea glass callback
glass(30.0, -0.1, 0.9, 2600); glass(30.35, -0.1, 0.5, 2600)

# vinyl crackle + gramophone colour for the whole piece
vc = np.zeros(N)
for k in range(int(DUR * 22)):
    i = int(rng.uniform(0, DUR - 0.01) * SR); vc[i:i + 120] += hp(rng.normal(0, 1, 120), 2500) * np.exp(-np.arange(120) / 18) * rng.uniform(0.05, 1)
vc += lp(rng.normal(0, 1, N), 4000) * 0.02

# ---------- mix ----------
mL = np.zeros(N); mR = np.zeros(N)
for bus, gcurve in (('band', band), ('cat', cat), ('night', night), ('free', np.ones(N))):
    bl, br = MUS[bus]; mL += bl * gcurve; mR += br * gcurve
def gramophone(x): return lp(hp(x, 120), 7500)
mL, mR = gramophone(mL), gramophone(mR)
# black-frame dip: low-pass the music around the fur transition
dip = curve([(0, 1), (16.9, 1), (17.4, 0.25), (17.9, 0.25), (18.3, 1), (35, 1)])
mL = mL * (0.5 + 0.5 * dip) ; mR = mR * (0.5 + 0.5 * dip)
irn = int(2.2 * SR); te = np.arange(irn) / SR; dec = np.exp(-te * 2.8)
irL = lp(rng.normal(0, 1, irn), 5000) * dec; irR = lp(rng.normal(0, 1, irn), 5000) * dec
irL /= np.sqrt((irL ** 2).sum()); irR /= np.sqrt((irR ** 2).sum())
mL = mL + 0.32 * oaconvolve(mL, irL)[:N]; mR = mR + 0.32 * oaconvolve(mR, irR)[:N]
sL = L + 0.18 * oaconvolve(L, irL)[:N]; sR = R + 0.18 * oaconvolve(R, irR)[:N]
outL = mL * 1.0 + sL * 1.0 + vc * 0.35; outR = mR * 1.0 + sR * 1.0 + vc * 0.35
st = np.stack([outL, outR], 1)
fo = int(2.2 * SR); st[-fo:] *= np.linspace(1, 0, fo)[:, None] ** 1.3
st[:int(0.15 * SR)] *= np.linspace(0, 1, int(0.15 * SR))[:, None]
st = np.tanh(st / np.abs(st).max() * 1.2) / np.tanh(1.2) * 0.9
wavfile.write(sys.argv[1], SR, (st * 32767).astype(np.int16))
print('ok', st.shape[0] / SR)
