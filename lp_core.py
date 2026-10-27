"""Inti pemrosesan Linear Prediction (tanpa GUI)."""
import io
import numpy as np


def load_signal(source):
    """source: path (str) atau bytes. Kolom 2 = amplitudo bila ada 2+ kolom,
    kalau 1 kolom dianggap amplitudo saja. Return (t, x)."""
    if isinstance(source, (bytes, bytearray)):
        text = source.decode("utf-8", errors="ignore")
    else:
        with open(source, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    text = text.replace(",", " ").replace(";", " ")
    rows = []
    for line in io.StringIO(text):
        parts = line.split()
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            continue  # lewati header / baris non-numerik
    if not rows:
        raise ValueError("Tidak ada data numerik pada file.")
    width = min(len(r) for r in rows)
    d = np.array([r[:width] for r in rows])
    if width == 1:
        return np.arange(len(d), dtype=float), d[:, 0]
    return d[:, 0], d[:, 1]


def autocorrelation(x, L):
    N = len(x)
    return np.array([np.dot(x[:N - k], x[k:]) / N for k in range(L + 1)])


def generate_matrix(r, L):
    idx = np.abs(np.subtract.outer(np.arange(L), np.arange(L)))
    return r[idx], r[1:L + 1].reshape(-1, 1)


def predict_signal(x, a):
    L = len(a)
    xp = np.zeros_like(x)
    for n in range(L, len(x)):
        xp[n] = np.dot(a, x[n - L:n][::-1])
    return xp


def frequency_response(a, G, nfft=512):
    w = np.linspace(0, np.pi, nfft)
    k = np.arange(1, len(a) + 1)
    A = 1 - np.exp(-1j * np.outer(w, k)) @ a
    return w, np.abs(G / A), np.abs(A)


def run_all(x, L):
    r = autocorrelation(x, L)
    Rxx, rv = generate_matrix(r, L)
    Rinv = np.linalg.inv(Rxx)
    a = (Rinv @ rv).ravel()
    xp = predict_signal(x, a)
    e = x - xp
    e[:L] = 0.0
    mse = float(np.mean(e[L:] ** 2))
    Em = float(r[0] - np.dot(a, r[1:L + 1]))
    G = float(np.sqrt(max(Em, 1e-30)))
    w, H, A = frequency_response(a, G)
    snr = float(10 * np.log10(np.mean(x[L:] ** 2) / mse)) if mse > 0 else float("inf")
    return dict(L=L, r=r, Rxx=Rxx, rv=rv, Rinv=Rinv, a=a, xp=xp, e=e,
                mse=mse, rms=mse ** 0.5, Emodel=Em, G=G, snr=snr,
                cond=float(np.linalg.cond(Rxx)), w=w, H=H, A=A)
