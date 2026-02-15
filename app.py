import streamlit as st
import cv2
import numpy as np
from PIL import Image
from mistralai import Mistral
import base64
import re
import pandas as pd
from groq import Groq
import os
from dotenv import load_dotenv
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import json

load_dotenv()

# Initialize API clients
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Optional

if MISTRAL_API_KEY:
    mistral_client = Mistral(api_key=MISTRAL_API_KEY)
else:
    st.warning("MISTRAL_API_KEY not set. Some features may be limited.")

if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
else:
    st.warning("GROQ_API_KEY not set. Some features may be limited.")

# Define expense categories
CATEGORIES = {
    'FOOD & DINING': ['food', 'restaurant', 'lunch', 'dinner', 'breakfast', 'cafe', 'coffee', 'pizza', 'burger', 'pasta', 'sushi', 'takeaway'],
    'GROCERIES': ['grocery', 'supermarket', 'vegetables', 'fruits', 'milk', 'bread', 'eggs', 'meat', 'dairy'],
    'TRANSPORTATION': ['transport', 'taxi', 'uber', 'lyft', 'bus', 'train', 'subway', 'fuel', 'petrol', 'gas', 'parking', 'toll'],
    'UTILITIES': ['utility', 'electricity', 'water', 'gas', 'internet', 'phone', 'bill', 'broadband', 'mobile'],
    'ENTERTAINMENT': ['entertainment', 'movie', 'cinema', 'netflix', 'spotify', 'game', 'sports', 'concert', 'theatre'],
    'SHOPPING': ['shopping', 'clothes', 'apparel', 'electronics', 'gadgets', 'accessories', 'amazon', 'walmart', 'target'],
    'HEALTHCARE': ['health', 'medical', 'medicine', 'doctor', 'hospital', 'pharmacy', 'dental', 'insurance'],
    'HOUSING': ['rent', 'mortgage', 'maintenance', 'repair', 'furniture', 'home'],
    'EDUCATION': ['education', 'tuition', 'book', 'course', 'training', 'workshop', 'school'],
    'PERSONAL CARE': ['salon', 'spa', 'gym', 'fitness', 'beauty', 'cosmetics', 'haircut'],
    'OTHER': []
}

# Financial advice prompts
FINANCIAL_ADVICE_PROMPT = """
You are a professional financial advisor. Based on the following expense data, provide personalized financial insights and recommendations.

Expense Summary:
- Total Spending: ${total_spending:.2f}
- Number of Transactions: {transaction_count}
- Date Range: {date_range}

Category Breakdown:
{category_breakdown}

Top Expenses:
{top_expenses}

Please provide:
1. **Executive Summary**: Brief overview of spending patterns
2. **Category Analysis**: Detailed breakdown of each major category with savings opportunities
3. **Budget Recommendations**: Specific budget suggestions for each category
4. **Smart Savings Tips**: 3-5 actionable tips based on spending habits
5. **Goal Setting**: Realistic financial goals for the next month
6. **Risk Alerts**: Any concerning spending patterns or potential issues

Format the response in a clear, professional manner with sections. Be specific and actionable with your advice.
"""

def preprocess_image(image):
    """Apply preprocessing pipeline to image"""
    img_array = np.array(image)
    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Noise Reduction
    denoise = cv2.GaussianBlur(img, (5,5), 0)
    
    # Grayscale
    gray = cv2.cvtColor(denoise, cv2.COLOR_BGR2GRAY)
    
    # Contrast Enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast = clahe.apply(gray)
    
    # Thresholding
    _, thresh = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return {
        'original': image,
        'grayscale': gray,
        'contrast': contrast,
        'threshold': thresh,
        'preprocessed_for_ocr': thresh
    }

def perform_ocr_mistral(image_array):
    """Perform OCR using Mistral AI"""
    try:
        if not MISTRAL_API_KEY:
            return "Mistral API key not configured"
            
        pil_image = Image.fromarray(image_array)
        
        import io
        img_bytes = io.BytesIO()
        pil_image.save(img_bytes, format='PNG')
        img_bytes = img_bytes.getvalue()
        
        base64_image = base64.b64encode(img_bytes).decode('utf-8')
        
        ocr_response = mistral_client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "image_url",
                "image_url": f"data:image/png;base64,{base64_image}"
            }
        )
        
        if ocr_response and ocr_response.pages:
            extracted_text = ""
            for page in ocr_response.pages:
                if hasattr(page, 'markdown'):
                    extracted_text += page.markdown + "\n"
            return extracted_text
        
        return "No text extracted"
    except Exception as e:
        st.error(f"OCR Error: {str(e)}")
        return None

def perform_ocr_groq(image_array):
    """Perform OCR using Groq's Llama vision model"""
    try:
        if not GROQ_API_KEY:
            return "Groq API key not configured"
            
        pil_image = Image.fromarray(image_array)
        
        import io
        img_bytes = io.BytesIO()
        pil_image.save(img_bytes, format='PNG')
        img_bytes = img_bytes.getvalue()
        
        base64_image = base64.b64encode(img_bytes).decode('utf-8')
        
        completion = groq_client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract all text from this receipt or document. Return only the extracted text, no additional explanations."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.1,
            max_tokens=1024
        )
        
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"Groq OCR Error: {str(e)}")
        return None

def parse_receipt_text(ocr_text):
    """Parse OCR text to extract structured data (items, prices, quantities)"""
    if not ocr_text:
        return []
    
    lines = ocr_text.split('\n')
    items = []
    
    # Regular expressions for price patterns
    price_patterns = [
        r'\$?\s*(\d+\.?\d*)\s*$',
        r'\$?\s*(\d+\.?\d*)\s*(?=\s|$)',
        r'(\d+\.\d{2})',
    ]
    
    for line in lines:
        line = line.strip()
        if not line or len(line) < 3:
            continue
            
        # Skip lines that look like headers/footers
        if any(keyword in line.lower() for keyword in ['total', 'tax', 'subtotal', 'change', 'cash', 'card', 'receipt', 'thank']):
            continue
            
        # Try to find price in the line
        price = None
        for pattern in price_patterns:
            price_match = re.search(pattern, line)
            if price_match:
                try:
                    price = float(price_match.group(1))
                    break
                except:
                    continue
        
        # Remove price from line to get item name
        if price:
            item_name = re.sub(price_patterns[0], '', line).strip()
            item_name = re.sub(r'[$]', '', item_name).strip()
            
            # Try to extract quantity (if available)
            quantity = 1
            quantity_match = re.search(r'(\d+)\s*x\s*', item_name, re.IGNORECASE)
            if quantity_match:
                quantity = int(quantity_match.group(1))
                item_name = re.sub(r'\d+\s*x\s*', '', item_name, flags=re.IGNORECASE).strip()
            
            if item_name and price > 0:
                items.append({
                    'name': item_name,
                    'price': round(price, 2),
                    'quantity': quantity,
                    'total': round(price * quantity, 2)
                })
    
    return items

def categorize_item(item_name):
    """Categorize an item based on its name"""
    item_name_lower = item_name.lower()
    
    for category, keywords in CATEGORIES.items():
        for keyword in keywords:
            if keyword in item_name_lower:
                return category
    
    return 'OTHER'

def analyze_spending(items):
    """Analyze spending by category and calculate statistics"""
    if not items:
        return None
    
    df = pd.DataFrame(items)
    
    # Add category column
    df['category'] = df['name'].apply(categorize_item)
    
    # Calculate category totals
    category_totals = df.groupby('category')['total'].sum().reset_index()
    category_totals = category_totals.sort_values('total', ascending=False)
    
    # Calculate overall statistics
    total_spending = df['total'].sum()
    category_totals['percentage'] = (category_totals['total'] / total_spending * 100).round(2)
    
    # Identify potential anomalies (items with unusually high prices)
    mean_price = df['price'].mean()
    std_price = df['price'].std()
    df['is_anomaly'] = df['price'] > (mean_price + 2 * std_price) if std_price > 0 else False
    
    # Calculate spending velocity (average per item)
    avg_price_per_item = df['price'].mean()
    
    return {
        'items_df': df,
        'category_totals': category_totals,
        'total_spending': total_spending,
        'anomalies': df[df['is_anomaly']],
        'avg_price_per_item': avg_price_per_item,
        'transaction_count': len(df),
        'unique_categories': len(category_totals)
    }

def generate_financial_advice(analysis_data, llm_provider="groq"):
    """Generate personalized financial advice using LLM"""
    
    if not analysis_data:
        return "No data available for analysis"
    
    # Prepare category breakdown for the prompt
    category_breakdown = ""
    for _, row in analysis_data['category_totals'].iterrows():
        category_breakdown += f"- {row['category']}: ${row['total']:.2f} ({row['percentage']}%)\n"
    
    # Prepare top expenses
    top_expenses = ""
    top_items = analysis_data['items_df'].nlargest(5, 'total')[['name', 'total', 'category']]
    for _, row in top_items.iterrows():
        top_expenses += f"- {row['name']}: ${row['total']:.2f} ({row['category']})\n"
    
    # Format the prompt
    prompt = FINANCIAL_ADVICE_PROMPT.format(
        total_spending=analysis_data['total_spending'],
        transaction_count=analysis_data['transaction_count'],
        date_range=datetime.now().strftime("%B %Y"),  # Current month
        category_breakdown=category_breakdown,
        top_expenses=top_expenses
    )
    
    try:
        if llm_provider == "groq" and GROQ_API_KEY:
            response = groq_client.chat.completions.create(
                model="mixtral-8x7b-32768",  # Using Mixtral for better financial advice
                messages=[
                    {"role": "system", "content": "You are an expert financial advisor providing personalized budgeting advice."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2048
            )
            return response.choices[0].message.content
            
        elif llm_provider == "mistral" and MISTRAL_API_KEY:
            response = mistral_client.chat.complete(
                model="mistral-large-latest",
                messages=[
                    {"role": "system", "content": "You are an expert financial advisor providing personalized budgeting advice."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2048
            )
            return response.choices[0].message.content
            
        else:
            # Fallback to rule-based advice if no LLM available
            return generate_rule_based_advice(analysis_data)
            
    except Exception as e:
        st.error(f"Error generating financial advice: {str(e)}")
        return generate_rule_based_advice(analysis_data)

def generate_rule_based_advice(analysis_data):
    """Generate rule-based financial advice as fallback"""
    
    advice = "📊 **Financial Insights (Rule-Based Analysis)**\n\n"
    
    # Total spending assessment
    total = analysis_data['total_spending']
    if total > 500:
        advice += f"⚠️ **High Spending Alert**: Your total spending (${total:.2f}) is relatively high. Consider reviewing your expenses.\n\n"
    else:
        advice += f"✅ Your total spending (${total:.2f}) is within a reasonable range.\n\n"
    
    # Category-based advice
    advice += "**Category Recommendations:**\n"
    
    for _, row in analysis_data['category_totals'].iterrows():
        category = row['category']
        amount = row['total']
        percentage = row['percentage']
        
        if category == 'FOOD & DINING' and percentage > 30:
            advice += f"- 🍽️ Food expenses are {percentage}% of total. Consider meal prepping to save money.\n"
        elif category == 'ENTERTAINMENT' and percentage > 20:
            advice += f"- 🎬 Entertainment is {percentage}% of total. Look for free activities or reduce subscription services.\n"
        elif category == 'SHOPPING' and percentage > 25:
            advice += f"- 🛍️ Shopping expenses are high at {percentage}%. Try the 24-hour rule before non-essential purchases.\n"
        elif category == 'TRANSPORTATION' and percentage > 20:
            advice += f"- 🚗 Transportation costs are {percentage}% of total. Consider carpooling or public transport.\n"
        else:
            advice += f"- {category}: ${amount:.2f} ({percentage}%) - Within normal range.\n"
    
    # Anomaly detection
    if not analysis_data['anomalies'].empty:
        advice += "\n**⚠️ Unusual Expenses Detected:**\n"
        for _, row in analysis_data['anomalies'].iterrows():
            advice += f"- {row['name']}: ${row['price']:.2f} is higher than average\n"
    
    # Savings tips
    advice += "\n**💡 Quick Savings Tips:**\n"
    advice += "1. Review recurring subscriptions\n"
    advice += "2. Use cashback apps for regular purchases\n"
    advice += "3. Set up automatic transfers to savings\n"
    advice += "4. Compare prices before major purchases\n"
    advice += "5. Consider the 50/30/20 budgeting rule\n"
    
    return advice

def main():
    st.set_page_config(
        page_title="Smart Receipt Scanner",
        page_icon="🧾",
        layout="wide"
    )
    
    st.title("🧾 Smart Receipt Scanner & Financial Advisor")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Settings")
        
        ocr_engine = st.selectbox(
            "OCR Engine",
            ["Groq (Llama Vision)", "Mistral AI"],
            help="Select which AI service to use for text extraction"
        )
        
        llm_provider = st.selectbox(
            "Financial Advisor AI",
            ["Groq (Mixtral)", "Mistral AI", "Rule-Based (Offline)"],
            help="Select AI for personalized financial advice"
        )
        
        st.header("📁 Categories")
        with st.expander("View Categories"):
            for category in CATEGORIES.keys():
                st.write(f"- {category}")
        
        # Budget goal setting
        st.header("🎯 Budget Goals")
        monthly_budget = st.number_input(
            "Monthly Budget Target ($)",
            min_value=0.0,
            value=2000.0,
            step=100.0
        )
        
        if 'total_spending' in st.session_state:
            remaining = monthly_budget - st.session_state.get('total_spending', 0)
            if remaining > 0:
                st.success(f"Remaining: ${remaining:.2f}")
            else:
                st.error(f"Over budget by: ${abs(remaining):.2f}")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Receipt or Document Image",
        type=["jpg", "jpeg", "png", "webp"]
    )
    
    if uploaded_file is not None:
        # Load and preprocess image
        image = Image.open(uploaded_file).convert("RGB")
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📸 Image Processing", 
            "🔤 OCR Results", 
            "📊 Parsed Data", 
            "💰 Spending Analysis",
            "🤖 Financial Advice"
        ])
        
        with tab1:
            st.subheader("Image Preprocessing Pipeline")
            
            # Apply preprocessing
            processed_images = preprocess_image(image)
            st.session_state['processed_images'] = processed_images
            
            # Display results in columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(processed_images['original'], caption="Original", use_column_width=True)
                st.image(processed_images['grayscale'], caption="Grayscale", channels="GRAY", use_column_width=True)
            
            with col2:
                st.image(processed_images['contrast'], caption="Contrast Enhanced", channels="GRAY", use_column_width=True)
                st.image(processed_images['threshold'], caption="Thresholded (for OCR)", channels="GRAY", use_column_width=True)
        
        with tab2:
            st.subheader("OCR Text Extraction")
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                if st.button("🔍 Extract Text", type="primary", use_container_width=True):
                    with st.spinner("Performing OCR..."):
                        if "Groq" in ocr_engine:
                            ocr_text = perform_ocr_groq(processed_images['threshold'])
                        else:
                            ocr_text = perform_ocr_mistral(processed_images['threshold'])
                        
                        if ocr_text:
                            st.session_state['ocr_text'] = ocr_text
                            st.success("Text extracted successfully!")
                        else:
                            st.error("Failed to extract text.")
            
            with col2:
                if st.button("📋 Clear Results", use_container_width=True):
                    if 'ocr_text' in st.session_state:
                        del st.session_state['ocr_text']
                    if 'items' in st.session_state:
                        del st.session_state['items']
                    st.rerun()
            
            if 'ocr_text' in st.session_state:
                edited_text = st.text_area(
                    "Extracted Text (Editable)",
                    st.session_state['ocr_text'],
                    height=300,
                    key="ocr_text_editor"
                )
                
                # Update session state if text is edited
                if edited_text != st.session_state['ocr_text']:
                    st.session_state['ocr_text'] = edited_text
                
                st.info("You can edit the text above if needed, then go to the 'Parsed Data' tab.")
            else:
                st.info("Click 'Extract Text' to start OCR processing.")
        
        with tab3:
            st.subheader("Data Parsing & Structuring")
            
            if 'ocr_text' in st.session_state and st.session_state['ocr_text']:
                if st.button("📋 Parse Receipt Data", type="primary"):
                    with st.spinner("Parsing items..."):
                        items = parse_receipt_text(st.session_state['ocr_text'])
                        
                        if items:
                            st.session_state['items'] = items
                            st.success(f"Found {len(items)} items!")
                        else:
                            st.warning("No items could be parsed. Try editing the OCR text manually.")
                
                if 'items' in st.session_state:
                    df_display = pd.DataFrame(st.session_state['items'])
                    
                    st.subheader("Extracted Items")
                    edited_df = st.data_editor(
                        df_display,
                        column_config={
                            "name": st.column_config.TextColumn("Item Name", width="large"),
                            "quantity": st.column_config.NumberColumn("Quantity", min_value=1, step=1),
                            "price": st.column_config.NumberColumn("Price ($)", min_value=0.0, format="$%.2f"),
                            "total": st.column_config.NumberColumn("Total ($)", format="$%.2f", disabled=True)
                        },
                        hide_index=True,
                        use_container_width=True,
                        num_rows="dynamic"
                    )
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("✅ Update Items"):
                            # Recalculate totals
                            edited_df['total'] = edited_df['quantity'] * edited_df['price']
                            st.session_state['items'] = edited_df.to_dict('records')
                            st.success("Items updated!")
                    
                    with col2:
                        if st.button("🗑️ Clear All Items"):
                            del st.session_state['items']
                            st.rerun()
            else:
                st.info("Please extract text from the image first in the OCR tab.")
        
        with tab4:
            st.subheader("Spending Analysis")
            
            if 'items' in st.session_state and st.session_state['items']:
                # Perform analysis
                analysis = analyze_spending(st.session_state['items'])
                st.session_state['analysis'] = analysis
                st.session_state['total_spending'] = analysis['total_spending']
                
                # Display summary metrics in a row
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Items", analysis['transaction_count'])
                with col2:
                    st.metric("Total Spending", f"${analysis['total_spending']:.2f}")
                with col3:
                    st.metric("Categories", analysis['unique_categories'])
                with col4:
                    st.metric("Avg Item Price", f"${analysis['avg_price_per_item']:.2f}")
                
                # Category breakdown
                st.subheader("📊 Spending by Category")
                
                # Two columns for chart and table
                col1, col2 = st.columns([1.2, 0.8])
                
                with col1:
                    # Create a bar chart
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    categories = analysis['category_totals']['category']
                    totals = analysis['category_totals']['total']
                    percentages = analysis['category_totals']['percentage']
                    
                    bars = ax.bar(categories, totals, color=sns.color_palette("husl", len(categories)))
                    ax.set_xlabel('Category')
                    ax.set_ylabel('Amount ($)')
                    ax.set_title('Spending by Category')
                    ax.tick_params(axis='x', rotation=45)
                    
                    # Add value labels on bars
                    for bar, total, pct in zip(bars, totals, percentages):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                                f'${total:.0f}\n({pct:.1f}%)',
                                ha='center', va='bottom', fontsize=9)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                
                with col2:
                    # Category totals table with styling
                    styled_df = analysis['category_totals'].style.format({
                        'total': '${:.2f}',
                        'percentage': '{:.1f}%'
                    })
                    st.dataframe(styled_df, use_container_width=True)
                
                # Item breakdown by category
                st.subheader("📝 Detailed Item Breakdown")
                
                # Create tabs for each category
                category_tabs = st.tabs(analysis['category_totals']['category'].tolist())
                
                for tab, category in zip(category_tabs, analysis['category_totals']['category']):
                    with tab:
                        category_items = analysis['items_df'][analysis['items_df']['category'] == category]
                        category_total = category_items['total'].sum()
                        
                        st.write(f"**Total: ${category_total:.2f}**")
                        st.dataframe(
                            category_items[['name', 'quantity', 'price', 'total']].style.format({
                                'price': '${:.2f}',
                                'total': '${:.2f}'
                            }),
                            use_container_width=True,
                            hide_index=True
                        )
                
                # Anomaly detection
                if not analysis['anomalies'].empty:
                    st.subheader("⚠️ Anomaly Detection")
                    st.warning("The following items are unusually expensive:")
                    
                    for _, row in analysis['anomalies'].iterrows():
                        st.write(f"- **{row['name']}**: ${row['price']:.2f} (Category: {row['category']})")
            else:
                st.info("Please parse receipt data first in the Parsed Data tab.")
        
        with tab5:
            st.subheader("🤖 AI Financial Advisor")
            
            if 'analysis' in st.session_state and st.session_state['analysis']:
                # LLM Provider mapping
                provider_map = {
                    "Groq (Mixtral)": "groq",
                    "Mistral AI": "mistral",
                    "Rule-Based (Offline)": "rule"
                }
                
                selected_provider = provider_map[llm_provider]
                
                # Add context about the receipt
                st.markdown("""
                <style>
                .advice-box {
                    padding: 20px;
                    border-radius: 10px;
                    background-color: purple;
                    margin: 10px 0px;
                }
                </style>
                """, unsafe_allow_html=True)
                
                # Button to generate advice
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    if st.button("💡 Generate Financial Advice", type="primary", use_container_width=True):
                        with st.spinner("AI is analyzing your spending patterns..."):
                            advice = generate_financial_advice(
                                st.session_state['analysis'],
                                selected_provider
                            )
                            st.session_state['financial_advice'] = advice
                
                with col2:
                    if st.button("📧 Email Advice", use_container_width=True):
                        st.info("Email feature coming soon!")
                
                with col3:
                    if st.button("📥 Save as PDF", use_container_width=True):
                        st.info("PDF export coming soon!")
                
                # Display financial advice
                if 'financial_advice' in st.session_state:
                    st.markdown("---")
                    
                    # Create a nice container for advice
                    with st.container():
                        st.markdown("### 📋 Personalized Financial Recommendations")
                        st.markdown("Based on your spending patterns, here's your customized advice:")
                        
                        # Display advice in a styled box
                        st.markdown(
                            f"""
                            <div class="advice-box">
                            {st.session_state['financial_advice'].replace(chr(10), '<br>')}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    # Quick action items based on advice
                    st.markdown("---")
                    st.markdown("### 🎯 Quick Action Items")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**📉 Reduce Expenses**")
                        st.markdown("- Cancel unused subscriptions")
                        st.markdown("- Use cashback apps")
                        st.markdown("- Compare prices online")
                    
                    with col2:
                        st.markdown("**💰 Savings Goals**")
                        st.markdown("- Set up automatic transfers")
                        st.markdown("- Save 20% of income")
                        st.markdown("- Build emergency fund")
                    
                    with col3:
                        st.markdown("**📊 Track Progress**")
                        st.markdown("- Review expenses weekly")
                        st.markdown("- Use budgeting apps")
                        st.markdown("- Set spending alerts")
                    
                    # Download advice as text
                    st.download_button(
                        label="📥 Download Advice",
                        data=st.session_state['financial_advice'],
                        file_name=f"financial_advice_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
            else:
                st.info("Please analyze your spending first in the 'Spending Analysis' tab.")

if __name__ == "__main__":
    main()"GRAY")
