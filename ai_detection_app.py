import streamlit as st
from PIL import Image
import random

# ---------- SAYFA AYARLARI ----------
st.set_page_config(page_title="AI Detection", page_icon="🤖", layout="wide")

# ---------- LOGO ----------
try:
    logo = Image.open("logo.png")
    st.image(logo, width=150)
except FileNotFoundError:
    pass  # Logo yoksa hata verme

# ---------- ÜST BANNER ----------
st.markdown("""
<div style="background-color:#4a90e2; padding:15px; border-radius:10px; margin-bottom:15px;">
<h2 style="color:white; text-align:center; margin:0;">🤖 AI Detection Demo</h2>
<p style="color:white; text-align:center; margin:0;">Resim Yapay Zeka Tespiti</p>
</div>
""", unsafe_allow_html=True)

# ---------- ANALIZ GEÇMİŞİ ----------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------- DUMMY AI DETECTOR ----------
def simple_ai_detector(image):
    """Rastgele Yapay Zeka / İnsan tahmini"""
    if random.random() > 0.5:
        return "Yapay Zeka", round(random.uniform(60, 100), 2)
    else:
        return "İnsan", round(random.uniform(60, 100), 2)

# ---------- DOSYA YÜKLEME ----------
uploaded_file = st.file_uploader("📷 Resim yükle (jpg/jpeg/png)", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Yüklenen Görsel", use_container_width=True)

    with st.spinner("Analiz ediliyor..."):
        label, score = simple_ai_detector(image)

    st.subheader("🔍 Sonuç")
    st.success(f"{label}: {score}%")

    # Analiz geçmişine kaydet
    st.session_state.history.append({
        "filename": uploaded_file.name,
        "result": f"{label}: {score}%"
    })

# ---------- ANALIZ GEÇMİŞİ ----------
if st.session_state.history:
    st.markdown("---")
    st.subheader("🕒 Analiz Geçmişi")
    for i, entry in enumerate(reversed(st.session_state.history), 1):
        st.markdown(f"**{i}. {entry['filename']}**")
        st.write(f"- {entry['result']}")
