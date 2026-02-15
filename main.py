import streamlit as st
import cv2
import easyocr
import pandas as pd
import numpy as np
import re
import plotly.express as px

# ----------------------------
# INIT OCR
# ----------------------------
reader = easyocr.Reader(['en'])

# ----------------------------
# IMAGE PREPROCESSING
# ----------------------------
def preprocess_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5,5), 0)
    img = cv2.threshold(img, 0, 255,
                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return img

# ----------------------------
# OCR EXTRACTION
# ----------------------------
def run_ocr(image):
    result = reader.readtext(image, detail=0)
    return result

# ----------------------------
# PARSE TEXT → STRUCTURED DATA
# ----------------------------
def parse_receipt(lines):
    items = []
    
    for line in lines:
        match = re.findall(r'([A-Za-z ]+)\s+(\d+\.\d{2})', line)
        if match:
            for m in match:
                items.append({
                    "item": m[0].strip(),
                    "price": float(m[1])
                })

    df = pd.DataFrame(items)
    return df

# ----------------------------
# EXPENSE CATEGORIZATION
# ----------------------------
def categorize(item):
    item = item.lower()
    
    if any(x in item for x in ["milk", "cheese", "yogurt"]):
        return "Dairy"
    elif any(x in item for x in ["bread", "cake"]):
        return "Bakery"
    elif any(x in item for x in ["chicken", "beef", "meat"]):
        return "Meat"
    elif any(x in item for x in ["chips", "cola", "snack"]):
        return "Snacks"
    else:
        return "Other"

# ----------------------------
# ANALYTICS
# ----------------------------
def spending_analysis(df):
    df["category"] = df["item"].apply(categorize)

    summary = df.groupby("category")["price"].sum().reset_index()

    total = summary["price"].sum()
    summary["percentage"] = (summary["price"]/total)*100

    return summary, total

# ----------------------------
# FAKE LLM ADVICE (Hackathon Safe)
# Replace with Gemini/OpenAI later
# ----------------------------
def generate_advice(summary):
    high = summary.sort_values("percentage", ascending=False).iloc[0]

    advice = f"""
    💡 You spend most on {high['category']} ({high['percentage']:.1f}%).
    
    Suggestions:
    - Reduce spending in high category.
    - Plan weekly budget before shopping.
    - Track small purchases daily.
    """
    return advice

# ============================
# STREAMLIT UI
# ============================

st.title("🧾 AI Receipt Analyzer")

uploaded = st.file_uploader(
    "Upload Receipt", 
    type=["jpg","png","jpeg"]
)

if uploaded:

    file_bytes = np.asarray(bytearray(uploaded.read()),
                            dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, caption="Original Receipt")

    processed = preprocess_image(image)

    st.image(processed, caption="Processed Image")

    text = run_ocr(processed)

    st.subheader("OCR Text")
    st.write(text)

    df = parse_receipt(text)

    if len(df)==0:
        st.error("No items detected.")
    else:
        st.subheader("Extracted Items")
        st.dataframe(df)

        summary, total = spending_analysis(df)

        st.subheader("Category Breakdown")
        st.dataframe(summary)

        fig = px.pie(
            summary,
            values="price",
            names="category",
            title="Spending Distribution"
        )
        st.plotly_chart(fig)

        st.subheader("🤖 AI Budget Advice")
        st.write(generate_advice(summary))

        st.success(f"Total Spending: ${total:.2f}")
