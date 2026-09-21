import os
import sys
import pandas as pd
import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to system path to import preprocessing and prediction
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from preprocessing import preprocess_text
from predict import predict_news

# Set page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design
st.markdown("""
<style>
    /* Styling headers */
    .main-title {
        font-family: 'Inter', sans-serif;
        color: #1E293B;
        font-weight: 800;
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        color: #64748B;
        font-size: 1.25rem;
        margin-bottom: 2rem;
    }
    
    /* Premium card containers */
    .card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
    }
    
    /* Result styling */
    .real-news-card {
        background-color: #ECFDF5;
        border: 2px solid #059669;
        border-radius: 12px;
        padding: 24px;
        color: #065F46;
        text-align: center;
    }
    .fake-news-card {
        background-color: #FEF2F2;
        border: 2px solid #DC2626;
        border-radius: 12px;
        padding: 24px;
        color: #991B1B;
        text-align: center;
    }
    .result-title {
        font-size: 2.25rem;
        font-weight: 700;
        margin-bottom: 10px;
    }
    .confidence-score {
        font-size: 1.25rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# App Sidebar
st.sidebar.markdown("# 📰 System Settings")
st.sidebar.write("This tool uses Natural Language Processing and Machine Learning models to analyze structural features of news articles and determine their authenticity.")

# Load models safely
@st.cache_resource
def load_ml_assets():
    model_path = os.path.join("models", "model.pkl")
    vectorizer_path = os.path.join("models", "vectorizer.pkl")
    
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer
    return None, None

model, vectorizer = load_ml_assets()

# Sidebar asset check
if model is None:
    st.sidebar.warning("⚠️ Warning: Model assets not found! Please run the training pipeline (`src/train_model.py`) first.")
else:
    st.sidebar.success("✅ Model and Vectorizer loaded successfully.")

# Main Application Layout
st.markdown('<div class="main-title">Fake News Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">An advanced NLP platform for verifying news authenticity using Machine Learning.</div>', unsafe_allow_html=True)

# Set up tabs
tab1, tab2 = st.tabs(["🔍 Predict News", "📊 Model Comparison & Analytics"])

with tab1:
    st.markdown("### Test Custom News Article")
    st.write("Input the headline and text of any article below. The system will clean the text, lemmatize it, extract TF-IDF features, and predict whether it is Real or Fake.")
    
    # Text input forms
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        title_input = st.text_input("News Headline", placeholder="Enter the article headline here...")
        text_input = st.text_area("News Body Text", placeholder="Paste the full body text of the article here...", height=250)
        
        predict_button = st.button("Run Prediction Pipeline", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        if predict_button:
            if not title_input.strip() and not text_input.strip():
                st.error("Please enter a headline or article body text to analyze.")
            elif model is None:
                st.error("No model found. Please run the model training first to save the model assets.")
            else:
                with st.spinner("Analyzing text patterns, calculating lexical weights..."):
                    # Combine title and text
                    full_text = f"{title_input} {text_input}"
                    
                    # Call predictions
                    pred_label, label_text, confidence = predict_news(full_text, model, vectorizer)
                    
                    if pred_label is None:
                        st.error("The input text was not valid after preprocessing (possibly only punctuation or empty).")
                    else:
                        st.subheader("Analysis Verdict")
                        
                        # Custom styled card for output
                        if pred_label == 1:
                            st.markdown(f"""
                            <div class="real-news-card">
                                <div class="result-title">🟢 {label_text}</div>
                                <div class="confidence-score">Confidence Score: {confidence * 100:.2f}%</div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="fake-news-card">
                                <div class="result-title">🔴 {label_text}</div>
                                <div class="confidence-score">Confidence Score: {confidence * 100:.2f}%</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Detailed Analysis progress bar
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown("**Probability Distribution:**")
                        st.progress(float(confidence))
                        st.write(f"The model is {confidence * 100:.1f}% confident that this news is **{label_text.replace(' NEWS', '')}**.")

with tab2:
    st.markdown("### Model Comparison & Evaluation Dashboard")
    st.write("Below are the experimental results from training various machine learning classifiers on the balanced ISOT dataset (15,000 articles, 80/20 train/test split).")
    
    # Check if comparison results exist
    comp_path = os.path.join("models", "comparison_results.csv")
    if os.path.exists(comp_path):
        # Load and show comparison table
        comp_df = pd.read_csv(comp_path)
        
        c1, c2 = st.columns([3, 2])
        
        with c1:
            st.markdown("#### Performance Metrics")
            st.dataframe(comp_df, width="stretch")
            
            # Matplotlib comparison plot
            st.markdown("#### F1-Score Comparison")
            # Convert f1-score string to float if needed
            comp_df['F1-Score_val'] = comp_df['F1-Score'].astype(float)
            fig, ax = plt.subplots(figsize=(6, 3))
            sns.barplot(
    x='F1-Score_val',
    y='Model',
    hue='Model',
    data=comp_df.sort_values(by='F1-Score_val', ascending=False),
    palette="Blues_r",
    legend=False,
    ax=ax
)
            ax.set_xlim(0, 1.0)
            ax.set_xlabel('F1-Score')
            ax.set_ylabel('')
            st.pyplot(fig)
            
        with c2:
            st.markdown("#### Confusion Matrix Visualizations")
            # Select model to show confusion matrix
            model_options = ["Logistic Regression", "Multinomial Naive Bayes", "Linear SVM", "Random Forest"]
            selected_model = st.selectbox("Select Model to View Confusion Matrix:", model_options)
            
            cm_filename = f"confusion_matrix_{selected_model.lower().replace(' ', '_')}.png"
            cm_path = os.path.join("notebook", "plots", cm_filename)
            
            if os.path.exists(cm_path):
                st.image(cm_path, caption=f"Confusion Matrix for {selected_model}", width="stretch")
            else:
                st.info("Confusion matrix image not found. Run training script to generate it.")
    else:
        st.info("No training results found yet. Run the training script first to populate this analytics dashboard.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #94A3B8;'>Fake News Detection System | Built with Streamlit, Scikit-learn, and NLTK</p>", unsafe_allow_html=True)
