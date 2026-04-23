import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import gdown
import os

if not os.path.exists("resnet50_model.pth"):
    url = "https://drive.google.com/uc?id=1tep_BS0wdHBist87i32H_Nw564jTvvw1"
    gdown.download(url, "resnet50_model.pth", quiet=False)

st.set_page_config(page_title="Plant Disease Detector", layout="centered")

# CSS
st.markdown("""
    <style>
    .main { text-align: center; }
    .stFileUploader {
        border: 2px dashed #4CAF50;
        padding: 20px;
        border-radius: 15px;
        background-color: #f9fff9;
    }
    </style>
""", unsafe_allow_html=True)

# Model
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 4)
model.load_state_dict(torch.load("resnet50_model.pth", map_location=torch.device('cpu')))
model.eval()

# ✅ ADD THIS (IMPORTANT)
classes = [
    'diseased cotton leaf',
    'diseased cotton plant',
    'fresh cotton leaf',
    'fresh cotton plant'
]

# Disease info
disease_info = {
    "diseased cotton leaf": {
        "description": "Leaf shows signs of infection such as spots, discoloration or damage.",
        "solution": "Remove affected leaves and apply appropriate fungicide or pesticide."
    },
    "diseased cotton plant": {
        "description": "The plant is infected and may show stunted growth or discoloration.",
        "solution": "Use disease-resistant seeds and apply proper treatment like pesticides."
    },
    "fresh cotton leaf": {
        "description": "Healthy leaf with no visible disease symptoms.",
        "solution": "No action needed. Maintain proper care and watering."
    },
    "fresh cotton plant": {
        "description": "Healthy plant with normal growth.",
        "solution": "Keep monitoring and maintain proper nutrients and irrigation."
    }
}

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Title
st.markdown("<h1>🌿 Plant Disease Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size:18px;'>Upload a leaf image to detect disease</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Please upload the plant image", type=["jpg", "png", "jpeg"])

# ✅ EVERYTHING INSIDE THIS BLOCK
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Uploaded Image", use_container_width=True)

    img = transform(image).unsqueeze(0)

    with st.spinner("🔍 Analyzing Image..."):
        with torch.no_grad():
            outputs = model(img)
            probs = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

    prediction = classes[predicted.item()]
    info = disease_info[prediction]

    # Result
    st.markdown(f"""
        <div style="
            background-color:#e6ffe6;
            padding:20px;
            border-radius:10px;
            font-size:20px;
            font-weight:bold;
            color:#006600;">
            🌱 Prediction: {prediction}
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div style="margin-top:10px; font-size:18px;">
            📊 Confidence: {confidence.item()*100:.2f}%
        </div>
    """, unsafe_allow_html=True)

    # Disease info
    st.markdown("### 🧾 Disease Information")
    st.write(f"**Description:** {info['description']}")
    st.write(f"**Solution:** {info['solution']}")

# Sidebar
st.sidebar.title("ℹ️ About")
st.sidebar.info("This app uses ResNet50 CNN model to detect cotton plant diseases.")