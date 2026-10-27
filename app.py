import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from lp_core import load_signal, run_all

st.set_page_config(page_title="Linear Prediction PCG", page_icon="🫀", layout="wide")

TEAL, RED, GREEN, GRAY = "#0f766e", "#dc2626", "#16a34a", "#94a3b8"
SAMPLE = os.path.join(os.path.dirname(__file__), "data", "DataBiomodPCG.txt")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem;}
.hero {background: linear-gradient(120deg,#0f766e,#0e7490); padding: 1.2rem 1.5rem;
       border-radius: 14px; color: white; margin-bottom: 1rem;}
.hero h1 {margin:0; font-size: 1.8rem; color: white;}
.hero p {margin:.2rem 0 0; opacity:.9;}
div[data-testid="stMetric"] {background:#f1f5f9; padding:.7rem 1rem; border-radius:12px;}
</style>
<div class="hero"><h1>🫀 Linear Prediction Phonocardiogram</h1>
<p>Pemodelan all-pole (AR) sinyal bunyi jantung · Aminy Widinal Hartiningrum</p></div>
""", unsafe_allow_html=True)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Pengaturan")
    up = st.file_uploader("Upload data PCG (.txt / .csv)", type=["txt", "csv", "dat"],
                          help="Kolom 1 = waktu, kolom 2 = amplitudo (atau 1 kolom amplitudo saja).")
    L = st.number_input("Time lag (L)", min_value=1, max_value=200, value=4, step=1)
    fs_in = st.number_input("Frekuensi sampling fs (Hz)", min_value=1.0, value=1000.0, step=100.0,
                            help="Dipakai untuk sumbu waktu & frekuensi.")
    use_time = st.toggle("Sumbu X dalam detik", value=False)
    use_hz = st.toggle("Respon frekuensi dalam Hz", value=False)
    st.caption("Tanpa upload, aplikasi memakai data contoh DataBiomodPCG.txt.")

# ---------------- Data ----------------
try:
    if up is not None:
        t, x = load_signal(up.getvalue())
        src = up.name
    else:
        t, x = load_signal(SAMPLE)
        src = "DataBiomodPCG.txt (contoh)"
except Exception as ex:
    st.error(f"Gagal membaca file: {ex}")
    st.stop()

L = int(L)
if L >= len(x) // 2:
    st.error("L terlalu besar untuk panjang data.")
    st.stop()
try:
    R = run_all(x, L)
except np.linalg.LinAlgError:
    st.error("Matriks Rxx singular (tidak bisa di-invers). Coba L lebih kecil atau data lain.")
    st.stop()

n = np.arange(len(x))
xs = n / fs_in if use_time else n
xlabel = "Waktu (s)" if use_time else "Sampel"


def fig(h=3.0):
    f, ax = plt.subplots(figsize=(7, h))
    ax.grid(alpha=.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return f, ax


# ---------------- Ringkasan ----------------
st.caption(f"Sumber data: **{src}** · {len(x)} sampel")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Orde (L)", L)
c2.metric("MSE prediksi", f"{R['mse']:.3e}")
c3.metric("Error model (E)", f"{R['Emodel']:.3e}")
c4.metric("SNR prediksi", f"{R['snr']:.2f} dB")

tabs = st.tabs(["① Sinyal & Autokorelasi", "② Matriks Rxx", "③ Koefisien",
                "④ Error", "⑤ Prediksi", "⑥ Respon Frekuensi", "⑦ Pengaruh L"])

# 1
with tabs[0]:
    a1, a2 = st.columns(2)
    with a1:
        f, ax = fig(); ax.plot(xs, x, color="black", lw=.8)
        ax.axhline(0, color=GRAY, ls="--", lw=.6)
        ax.set(title="Sinyal input", xlabel=xlabel, ylabel="Amplitudo"); st.pyplot(f)
    with a2:
        f, ax = fig(); ax.stem(range(L + 1), R["r"]); ax.set(title="Autokorelasi r[k]", xlabel="k", ylabel="r[k]")
        st.pyplot(f)
    st.dataframe(pd.DataFrame({"k": range(L + 1), "r[k]": R["r"]}), hide_index=True, width="stretch")

# 2
with tabs[1]:
    idx = [str(i + 1) for i in range(L)]
    b1, b2 = st.columns([3, 1])
    b1.subheader("Matriks Rxx (Toeplitz)")
    b1.dataframe(pd.DataFrame(R["Rxx"], index=idx, columns=idx), width="stretch")
    b2.subheader("Vektor r")
    b2.dataframe(pd.DataFrame(R["rv"], index=idx, columns=["r"]), width="stretch")
    st.subheader("Inverse Rxx")
    st.dataframe(pd.DataFrame(R["Rinv"], index=idx, columns=idx), width="stretch")
    st.caption(f"Condition number Rxx = {R['cond']:.3g} (semakin besar → makin mendekati singular).")

# 3
with tabs[2]:
    st.latex(r"\mathbf{a}=\mathbf{R}_{xx}^{-1}\,\mathbf{r}")
    st.dataframe(pd.DataFrame({"Koefisien": [f"a{k+1}" for k in range(L)], "Nilai": R["a"]}),
                 hide_index=True, width="stretch")
    f, ax = fig(2.4); ax.bar(range(1, L + 1), R["a"], color=TEAL)
    ax.set(title="Koefisien prediktor", xlabel="k", ylabel="a[k]"); st.pyplot(f)

# 4
with tabs[3]:
    d1, d2 = st.columns([2, 1])
    with d1:
        f, ax = fig(); ax.plot(xs[L:], R["e"][L:], color="#2563eb", lw=.7)
        ax.set(title="Error prediksi e[n]", xlabel=xlabel, ylabel="e[n]"); st.pyplot(f)
    with d2:
        st.markdown("**Error prediksi (terukur)**")
        st.write(f"MSE = `{R['mse']:.6g}`  \nRMS = `{R['rms']:.6g}`  \nmax|e| = `{np.max(np.abs(R['e'])):.4g}`")
        st.markdown("**Error pemodelan (teoritis)**")
        st.latex(r"E=r_0-\sum a_k r_k")
        st.write(f"E = `{R['Emodel']:.6g}`  \nG = √E = `{R['G']:.6g}`")

# 5
with tabs[4]:
    f, ax = fig(3.6)
    ax.plot(xs, x, color=GRAY, lw=.9, label="Asli")
    ax.plot(xs[L:], R["xp"][L:], color=RED, lw=.8, label="Prediksi")
    ax.axhline(0, color=GRAY, ls="--", lw=.6)
    ax.set(title=f"Sinyal prediksi (L={L})", xlabel=xlabel, ylabel="Amplitudo"); ax.legend()
    st.pyplot(f)
    lo, hi = st.slider("Zoom (sampel)", 0, len(x) - 1, (0, min(500, len(x) - 1)))
    f, ax = fig(); sl = slice(lo, hi + 1)
    ax.plot(xs[sl], x[sl], color=GRAY, label="Asli")
    ax.plot(xs[sl], np.where(n[sl] >= L, R["xp"][sl], np.nan), color=RED, label="Prediksi")
    ax.legend(); ax.set(xlabel=xlabel); st.pyplot(f)

# 6
with tabs[5]:
    f, ax = fig(3.6)
    w = R["w"] * fs_in / (2 * np.pi) if use_hz else R["w"]
    ax.plot(w, R["H"], color=RED, label="H(ω) = G/A(ω)")
    ax.plot(w, R["A"], color=GREEN, label="A(ω)")
    ax.set(title="Respon frekuensi", xlabel="Frekuensi (Hz)" if use_hz else "ω (rad)", ylabel="Magnitudo")
    if not use_hz:
        ax.set_xticks([0, np.pi / 2, np.pi]); ax.set_xticklabels(["0", "π/2", "π"])
    ax.legend(); st.pyplot(f)

# 7
with tabs[6]:
    Lmax = st.slider("L maksimum", 5, 60, 30)
    rows = []
    for l in range(1, Lmax + 1):
        try:
            q = run_all(x, l); rows.append((l, q["mse"], q["Emodel"], q["snr"]))
        except np.linalg.LinAlgError:
            break
    df = pd.DataFrame(rows, columns=["L", "MSE", "E model", "SNR (dB)"])
    f, ax = fig(); ax.plot(df["L"], df["MSE"], "o-", color=TEAL)
    ax.axvline(L, color=RED, ls="--", lw=.8); ax.set(title="MSE terhadap orde L", xlabel="L", ylabel="MSE")
    st.pyplot(f)
    st.dataframe(df, hide_index=True, width="stretch")

# ---------------- Download ----------------
st.divider()
res = pd.DataFrame({"sampel": n, "waktu": t, "x": x, "x_prediksi": R["xp"], "error": R["e"]})
coef = pd.DataFrame({"koefisien": [f"a{k+1}" for k in range(L)], "nilai": R["a"]})
d1, d2 = st.columns(2)
d1.download_button("⬇️ Unduh sinyal prediksi & error (CSV)", res.to_csv(index=False).encode(),
                   "hasil_prediksi.csv", "text/csv")
d2.download_button("⬇️ Unduh koefisien prediktor (CSV)", coef.to_csv(index=False).encode(),
                   "koefisien.csv", "text/csv")
