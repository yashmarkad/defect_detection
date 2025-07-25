\# 🧠 Fraud + Defect Detection App (Streamlit + Deep Learning)



This project provides a dual-function AI system for:



\- ✅ \*\*Fraud Detection\*\*: Checks if a user-uploaded product image matches the reference catalog image using deep learning (ResNet50 + cosine similarity).

\- 🛠️ \*\*Defect Detection\*\*: Detects whether a product is defective using a custom-trained Deep Neural Network model in Keras.



It includes two applications:



\- `app2.py` — Upload product images manually from your system.

\- `app4.py` — Analyze product images directly from URL.



---



\## 📂 Project Structure



├── app2.py # Streamlit app for manual image uploads

├── app4.py # Streamlit app for image URLs

├── defect\_detection\_1.h5 # Trained DNN model for defect classification

├── requirements.txt # All required dependencies

└── README.md # Project documentation



---



\## 🖥️ Technologies Used



\- Python 3.11 (64-bit)

\- Streamlit 1.47.0

\- TensorFlow 2.15.0

\- Keras 2.15.0

\- rembg (for background removal)

\- ONNX Runtime 1.17.3

\- ResNet50 (from Keras Applications)

\- Scikit-learn (for cosine similarity)



---



\## 🔧 Setup Instructions



\### 1️⃣ Clone the Repository



```bash

git clone https://github.com/vibhutinile/ImageComparingTool.git





2️⃣ Create a Python 3.11 Environment 

python -m venv venv

venv\\Scripts\\activate  # On Windows

\# or

source venv/bin/activate  # On Linux/Mac



3️⃣ Install Dependencies

pip install -r requirements.txt





▶️ Running the Applications

Upload Images (Local):

streamlit run app2.py



Use Image URLs:

streamlit run app4.py

