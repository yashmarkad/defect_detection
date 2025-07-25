# ✅ Final version with improved logic for:
# 1. Showing all web comparisons for each user image
# 2. Defect checking per image
# 3. Final summary based on total frauds + defects

import streamlit as st
import requests
import io
import numpy as np
import ast
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from rembg import remove

# Page setup
st.set_page_config(page_title="Image Similarity + Defect Detection", layout="wide")

# Load models
@st.cache_resource
def load_models():
    resnet = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    defect_model = load_model("defect_detection_1.h5")
    return resnet, defect_model

resnet_model, defect_model = load_models()

# Helpers
def download_image_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        return img
    except Exception as e:
        st.warning(f"Failed to load image from {url}: {e}")
        return None

def remove_background(img):
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return Image.open(io.BytesIO(remove(buf.getvalue()))).convert("RGB")

def get_embedding(img):
    img = remove_background(img)
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = resnet_model.predict(x)
    return features[0]

def preprocess_for_defect(img):
    img = img.resize((224, 224))
    x = image.img_to_array(img) / 255.0
    return np.expand_dims(x, axis=0)

def predict_defect(img):
    x = preprocess_for_defect(img)
    pred = defect_model.predict(x)[0][0]
    return ("Defective", pred) if pred < 0.80 else ("Non-defective", pred)

# UI Input
st.sidebar.header("🔗 Enter URL Lists (Python format)")
web_urls_input = st.sidebar.text_area("🖼️ Web Image URLs", placeholder='["https://example.com/web1.jpg"]')
user_urls_input = st.sidebar.text_area("🏭 User Image URLs", placeholder='["https://example.com/user1.jpg"]')
threshold = st.sidebar.slider("🔍 Similarity Threshold", 0.5, 1.0, 0.85, 0.01)
start_button = st.sidebar.button("🚀 Start Analysis")

def parse_url_list(input_text, label):
    try:
        url_list = ast.literal_eval(input_text)
        if isinstance(url_list, list) and all(isinstance(url, str) for url in url_list):
            return url_list
        else:
            st.sidebar.error(f"❌ {label} must be a list of strings.")
            return []
    except Exception as e:
        st.sidebar.error(f"❌ Invalid list format for {label}: {e}")
        return []

# Run analysis
if start_button:
    web_urls = parse_url_list(web_urls_input, "Web URLs")
    user_urls = parse_url_list(user_urls_input, "User URLs")

    if not web_urls or not user_urls:
        st.warning("⚠️ Please enter valid Web and User URL lists.")
    else:
        web_data = []
        for url in web_urls:
            img = download_image_from_url(url)
            if img:
                emb = get_embedding(img)
                web_data.append((url, img, emb))

        if not web_data:
            st.error("❌ No valid Web images loaded.")
        else:
            total_defective = 0
            total_fraud = 0

            for user_url in user_urls:
                user_img = download_image_from_url(user_url)
                if not user_img:
                    continue

                user_emb = get_embedding(user_img)
                st.subheader(f"🏭 User Image: {user_url}")
                max_sim = 0
                match_found = False

                for web_url, web_img, web_emb in web_data:
                    sim = cosine_similarity([user_emb], [web_emb])[0][0]
                    st.markdown(f"🔗 Comparing with **{web_url}** — Similarity: `{sim:.2f}`")
                    col1, col2 = st.columns(2)
                    col1.image(user_img, caption="User Image", use_container_width=True)
                    col2.image(web_img, caption="Web Reference", use_container_width=True)
                    max_sim = max(max_sim, sim)

                    if sim >= threshold:
                        match_found = True

                if match_found:
                    st.success(f"✅ Match Found (Highest Similarity: {max_sim:.2f})")
                    label, score = predict_defect(user_img)
                    st.markdown(f"**🔬 Defect Prediction:** `{label}` (Confidence: {score:.2f})")
                    if label == "Defective":
                        total_defective += 1
                        st.error("❌ Defective Image")
                    else:
                        st.success("✅ Non-defective Image")
                else:
                    total_fraud += 1
                    st.warning("🚨 Fraud Alert: No matching image found.")

                st.divider()

            # Final summary
            st.header("🔍 Final Summary")
            if total_fraud == len(user_urls):
                st.error("🚨 FINAL VERDICT: FRAUD — All user images unmatched.")
            elif total_fraud > 0 and total_defective > 0:
                st.error(f"🚨 FINAL VERDICT: FRAUD + DEFECT — {total_fraud} frauds, {total_defective} defects.")
            elif total_fraud > 0:
                st.error(f"🚨 FINAL VERDICT: FRAUD — {total_fraud} fraudulent images.")
            elif total_defective >= 2:
                st.error(f"❌ FINAL VERDICT: DEFECTIVE — {total_defective} defective images.")
            else:
                st.success("✅ FINAL VERDICT: NON-DEFECTIVE — No major issues detected.")
