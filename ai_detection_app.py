import streamlit as st
from transformers import pipeline
from PIL import Image
import tempfile
import pandas as pd
import json
import time

# ---------- SAYFA AYARLARI ----------
st.set_page_config(page_title="AI Detection", page_icon="🤖", layout="wide")
st.markdown("""
<style>
.stApp {
    background-color: #f7f9fc;
}
</style>
""", unsafe_allow_html=True)

# ---------- LOGO ----------
logo_path = "logo.png"  # Repo köküne logo.png ekleyin
try:
    logo = Image.open(logo_path)
    st.image(logo, width=150)
except FileNotFoundError:
    st.write("🔹 Logo bulunamadı, lütfen 'logo.png' ekleyin.")

# ---------- ÜST BANNER ----------
st.markdown("""
<div style="background-color:#4a90e2; padding:15px; border-radius:10px; margin-bottom:15px;">
<h2 style="color:white; text-align:center; margin:0;">🤖 AI Detection Demo</h2>
<p style="color:white; text-align:center; margin:0;">Resim ve Video Yapay Zeka Tespiti</p>
</div>
""", unsafe_allow_html=True)

# ---------- MODELLER ----------
image_detector = pipeline("image-classification", model="umm-maybe/AI-image-detector", device=-1) 
video_detector = pipeline("video-classification", model="facebook/mvi-vit-base", device=-1)


# ---------- ETIKETLER ----------
image_label_mapping = {"artificial": "Yapay Zeka", "human": "İnsan"}
video_label_mapping = {"fake": "Yapay Zeka", "real": "İnsan"}

# ---------- ANALIZ GEÇMİŞİ ----------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------- SEÇİM ----------
option = st.radio("📌 Hangi tür içerik yükleyeceksiniz?", ("Resim", "Video"))

st.markdown("---")

# ---------- GRID SUTUNLARI ----------
col_left, col_right = st.columns(2)

# ---------- GIF LOADING ----------
loading_gif_path = "loading.gif"
try:
    loading_gif = Image.open(loading_gif_path)
except FileNotFoundError:
    loading_gif = None

# ---------- RESIM TESPITI ----------
if option == "Resim":
    uploaded_file = col_left.file_uploader("📷 Resim yükle (jpg/jpeg/png)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col_left.image(image, caption="Yüklenen Görsel", use_container_width=True)

        # Progress bar ve GIF
        progress_text = col_right.empty()
        progress_bar = col_right.progress(0)
        loading_placeholder = col_right.empty()

        with st.spinner("Analiz ediliyor..."):
            for i in range(1, 101):
                progress_text.text(f"Analiz ilerleme: {i}%")
                progress_bar.progress(i)
                if loading_gif:
                    loading_placeholder.image(loading_gif, width=80)
                time.sleep(0.02)
            results = image_detector(image)

        progress_text.text("Analiz tamamlandı ✅")
        progress_bar.empty()
        loading_placeholder.empty()

        # Sonuçlar
        display_results = []
        col_right.subheader("🔍 Sonuçlar")
        for r in results:
            label_en = r["label"]
            label_tr = image_label_mapping.get(label_en.lower(), label_en)
            score = round(r["score"] * 100, 2)
            col_right.success(f"{label_tr}: {score}%")
            display_results.append(f"{label_tr}: {score}%")

        st.session_state.history.append({
            "type": "Resim",
            "filename": uploaded_file.name,
            "results": display_results
        })

# ---------- VIDEO TESPITI ----------
elif option == "Video":
    uploaded_file = col_left.file_uploader("🎥 Video yükle (mp4/avi/mov)", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.flush()

        col_left.video(tfile.name)

        # Progress bar ve GIF
        progress_text = col_right.empty()
        progress_bar = col_right.progress(0)
        loading_placeholder = col_right.empty()

        with st.spinner("Analiz ediliyor... Lütfen bekleyin"):
            for i in range(1, 101):
                progress_text.text(f"Analiz ilerleme: {i}%")
                progress_bar.progress(i)
                if loading_gif:
                    loading_placeholder.image(loading_gif, width=80)
                time.sleep(0.02)
            try:
                results = video_detector(tfile.name)
            except Exception as e:
                col_right.error(f"Analiz sırasında bir hata oluştu: {e}")
                results = []

        progress_text.text("Analiz tamamlandı ✅")
        progress_bar.empty()
        loading_placeholder.empty()

        # Sonuçlar
        display_results = []
        col_right.subheader("📊 Sonuçlar")
        for r in results:
            label_en = r.get("label", "unknown")
            label_tr = video_label_mapping.get(label_en.lower(), label_en)
            score = r.get("score", None)
            if score is not None:
                col_right.success(f"{label_tr}: {round(score * 100, 2)}%")
                display_results.append(f"{label_tr}: {round(score * 100, 2)}%")
            else:
                col_right.warning(f"{label_tr}: -")
                display_results.append(f"{label_tr}: -")

        st.session_state.history.append({
            "type": "Video",
            "filename": uploaded_file.name,
            "results": display_results
        })

st.markdown("---")

# ---------- ANALIZ GEÇMİŞİ ----------
if st.session_state.history:
    st.subheader("🕒 Analiz Geçmişi")
    for i, entry in enumerate(reversed(st.session_state.history), 1):
        st.markdown(f"**{i}. {entry['type']} - {entry['filename']}**")
        for r in entry["results"]:
            st.write(f"- {r}")

    st.markdown("---")
    st.subheader("💾 Analiz Geçmişini İndir")

    # JSON
    json_data = json.dumps(st.session_state.history, ensure_ascii=False, indent=2)
    st.download_button(
        label="📥 JSON olarak indir",
        data=json_data,
        file_name="ai_detection_history.json",
        mime="application/json"
    )

    # CSV
    csv_rows = []
    for entry in st.session_state.history:
        csv_rows.append({
            "Tür": entry["type"],
            "Dosya Adı": entry["filename"],
            "Sonuçlar": "; ".join(entry["results"])
        })
    df = pd.DataFrame(csv_rows)
    csv_data = df.to_csv(index=False, sep=';', encoding='utf-8-sig')
    st.download_button(
        label="📥 CSV olarak indir",
        data=csv_data,
        file_name="ai_detection_history.csv",
        mime="text/csv"
    )


