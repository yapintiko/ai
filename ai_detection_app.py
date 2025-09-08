import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.models import resnet18

# ---------- SAYFA AYARLARI ----------
st.set_page_config(page_title="AI Detection", page_icon="🤖", layout="wide")

# ---------- LOGO ----------
try:
    logo = Image.open("logo.png")
    st.image(logo, width=150)
except FileNotFoundError:
    pass

# ---------- ÜST BANNER ----------
st.markdown("""
<div style="background-color:#4a90e2; padding:15px; border-radius:10px; margin-bottom:15px;">
<h2 style="color:white; text-align:center; margin:0;">🤖 AI Detection Demo</h2>
<p style="color:white; text-align:center; margin:0;">Resim Yapay Zeka Tespiti</p>
</div>
""", unsafe_allow_html=True)

# ---------- MODEL ----------
model = resnet18(pretrained=True)
model.eval()

# ---------- TRANSFORMS ----------
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# ---------- ANALIZ GEÇMİŞİ ----------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------- DOSYA YÜKLEME ----------
uploaded_file = st.file_uploader("📷 Resim yükle (jpg/jpeg/png)", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Yüklenen Görsel", use_container_width=True)

    with st.spinner("Analiz ediliyor..."):
        img_t = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img_t)
            _, preds = torch.max(outputs, 1)
        label = preds.item()

    st.subheader("🔍 Sonuç")
    st.success(f"Predicted class index: {label}")

    st.session_state.history.append({
        "filename": uploaded_file.name,
        "result": f"Predicted class index: {label}"
    })

# ---------- ANALIZ GEÇMİŞİ ----------
if st.session_state.history:
    st.markdown("---")
    st.subheader("🕒 Analiz Geçmişi")
    for i, entry in enumerate(reversed(st.session_state.history), 1):
        st.markdown(f"**{i}. {entry['filename']}**")
        st.write(f"- {entry['result']}")
