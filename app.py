import streamlit as st
from transformers import pipeline
import PyPDF2
from docx import Document
import pandas as pd
import torch
import time
import os

# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

st.set_page_config(
    page_title="HRVibeCheck",
    page_icon="👔",
    layout="wide"
)

st.markdown("""
    <style>
    .main-header { font-size: 2.8rem; font-weight: 700; color: #1e3a8a; margin-bottom: 0; }
    .sub-header { font-size: 1.1rem; color: #64748b; margin-bottom: 1.5rem; }
    .skill-pill { background-color: #3b82f6; color: white; padding: 6px 14px; border-radius: 30px;
                  margin: 4px; display: inline-block; font-weight: 500; font-size: 0.85rem; }
    </style>
    """, unsafe_allow_html=True)

# ==================== LOAD PIPELINES ====================
@st.cache_resource(show_spinner="Loading AI Models... This may take 20-40 seconds.")
def load_pipelines():
    # Pipeline 1: Your Latest Fine-tuned Model
    pipe1 = pipeline(
        "text-classification",
        model="Cheykong/HRVibeCheck-Hire-Recommendation-Model",
        device=0 if torch.cuda.is_available() else -1
    )
    
    # Pipeline 2: Skill Extraction
    pipe2 = pipeline(
        "token-classification",
        model="algiraldohe/lm-ner-linkedin-skills-recognition",
        aggregation_strategy="simple",
        device=0 if torch.cuda.is_available() else -1
    )
    return pipe1, pipe2

pipe1, pipe2 = load_pipelines()

# ==================== SIDEBAR DEBUG ====================
st.sidebar.success("✅ Model Loaded Successfully")
st.sidebar.write("**Loaded Model:**", pipe1.model.config._name_or_path)
st.sidebar.write("**Num Labels:**", pipe1.model.config.num_labels)

# ==================== HELPER FUNCTIONS ====================
def extract_text_from_file(uploaded_file):
    try:
        if uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
           
