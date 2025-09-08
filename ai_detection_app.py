import streamlit as st
from transformers import pipeline
from PIL import Image
import io

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
try:
    logo = Image.open("logo.png")
    st.image(logo, width=150)
except FileNotFoundError:
    pass  # Logo yoksa uyarı gösterme

# ---------- ÜST BANNER ----------
st.markdown("""
<div style="background-color:#4a90e2; padding:15px; border-radius:10px; margin-bottom:15px;">
<h2 style="color:white; text-align:center; margin:0;">🤖 AI Detection Demo</h2>
<p style="color:white; text-align:center; margin:0;">Resim Yapay Zeka Tespiti</p>
</div>
""", unsafe_allow_html=True)

# ---------- RESIM DETECTOR ----------
image_detector = pipeline(
    "image-classification",
    model="google/vit-base-patch16-224",
    device=-1  # CPU kullan
)

# ---------- ETIKETLER ----------
label_mapping = {"LABEL_0": "İnsan", "LABEL_1": "Yapay Zeka"}  # Modelin etiketleri farklı olabilir

# ---------- ANALIZ GEÇMİŞİ ----------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------- DOSYA YÜKLEME ----------
uploaded_file = st.file_uploader("📷 Resim yükle (jpg/jpeg/png)", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Yüklenen Görsel", use_container_width=True)

    with st.spinner("Analiz ediliyor..."):
        results = image_detector(image)

    st.subheader("🔍 Sonuçlar")
    display_results = []
    for r in results:
        label = r["label"]
        score = round(r["score"] * 100, 2)
        # Etiketleri Türkçeye çevir
        label_tr = label_mapping.get(label, label)
        st.success(f"{label_tr}: {score}%")
        display_results.append(f"{label_tr}: {score}%")

    st.session_state.history.append({
        "filename": uploaded_file.name,
        "results": display_results
    })

# ---------- ANALIZ GEÇMİŞİ ----------
if st.session_state.history:
    st.markdown("---")
    st.subheader("🕒 Analiz Geçmişi")
    for i, entry in enumerate(reversed(st.session_state.history), 1):
        st.markdown(f"**{i}. {entry['filename']}**")
        for r in entry["results"]:
            st.write(f"- {r}")
