import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Image Preprocessing Pipeline")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:

    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # RGB → BGR (OpenCV format)
    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # 1️⃣ Noise Reduction (Gaussian Blur)
    denoise = cv2.GaussianBlur(img, (5,5), 0)

    # 2️⃣ Grayscale Conversion
    gray = cv2.cvtColor(denoise, cv2.COLOR_BGR2GRAY)

    # 3️⃣ Contrast Enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast = clahe.apply(gray)

    # 4️⃣ Thresholding (Binary)
    _, thresh = cv2.threshold(
        contrast, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Display results
    st.image(image, caption="Original")
    st.image(gray, caption="Grayscale", channels="GRAY")
    st.image(contrast, caption="Contrast Enhanced", channels="GRAY")
    st.image(thresh, caption="Thresholded", channels="GRAY")
