import streamlit as st
from transformers import pipeline
from PIL import Image
import io

# Hugging Face AI detection modeli
detector = pipeline("image-classification", model="umm-maybe/AI-image-detector")

st.set_page_config(page_title="AI Detection Demo", page_icon="🤖", layout="centered")

st.title("🤖 AI Detection Demo")
st.write("Bir resim yükle, gerçek mi yoksa yapay zekâ ile mi üretilmiş öğren!")

uploaded_file = st.file_uploader("Resim yükle", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Görseli aç
    image = Image.open(uploaded_file)
    st.image(image, caption="Yüklenen Görsel", use_column_width=True)

    # AI Detection çalıştır
    with st.spinner("Analiz ediliyor..."):
        results = detector(image)

    st.subheader("🔍 Sonuç")
    for r in results:
        label = r["label"]
        score = round(r["score"] * 100, 2)
        st.write(f"**{label}**: {score}%")
