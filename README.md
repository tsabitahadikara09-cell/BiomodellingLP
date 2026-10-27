# Linear Prediction Phonocardiogram (Streamlit)

Aplikasi web untuk linear prediction pada sinyal PCG. Upload file data (kolom 1 = waktu, kolom 2 = amplitudo) lalu atur time lag L.

## Jalankan lokal
```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy lewat GitHub → Streamlit Community Cloud
1. Buat repo GitHub baru, upload seluruh isi folder ini (`app.py`, `lp_core.py`, `requirements.txt`, `data/`, `.streamlit/`).
2. Buka https://share.streamlit.io → login dengan GitHub → **Create app**.
3. Pilih repo, branch `main`, Main file path: `app.py` → **Deploy**.
4. Tunggu build selesai, link `https://<nama-app>.streamlit.app` siap dibagikan.
