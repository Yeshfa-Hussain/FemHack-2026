import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Upload Image with OpenCV")

# Upload image from user
uploaded_file = st.file_uploader("Choose an image", type=["jpg","jpeg","png","webp"])

if uploaded_file is not None:
    
    # Read image using PIL
    image = Image.open(uploaded_file)

    # Convert to numpy array
    img_array = np.array(image)

    # Convert RGB -> BGR (OpenCV uses BGR)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Example OpenCV processing (Gray image)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # Show original and processed images
    st.image(image, caption="Original Image")
    st.image(gray, caption="Gray Image (OpenCV)", channels="GRAY")
