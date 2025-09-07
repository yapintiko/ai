import streamlit as st
from transformers import pipeline
from PIL import Image

# Hugging Face modelini yükle
# Burada örnek olarak küçük bir AI-image detector modeli seçtik
detector = pipeline("image-classification", model="umm-maybe/AI-image-detector")

st.set_page_config(page_title="AI Image Detection", page_icon="🖼️", layout="centered")

st.title("🖼️ AI Image Detection Demo")
st.write("Bir resim yükle, yapay zekâ tarafından üretilmiş mi öğren!")

# Dosya yükleme alanı
uploaded_file = st.file_uploader("Resim yükle", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Görseli aç ve ekranda göster
    image = Image.open(uploaded_file)
    st.image(image, caption="Yüklenen Görsel", use_column_width=True)

    # Analiz yap
    with st.spinner("Analiz ediliyor..."):
        results = detector(image)

    # Sonucu göster
    st.subheader("🔍 Sonuç")
    for r in results:
        label = r["label"]
        score = round(r["score"] * 100, 2)
        st.write(f"**{label}**: {score}%")
