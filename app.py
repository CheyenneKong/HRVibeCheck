import streamlit as st
from transformers import pipeline
import PyPDF2
from docx import Document

# --- PAGE CONFIG ---
st.set_page_config(page_title="HRVibeCheck", page_icon="👔", layout="wide")

# HR-Friendly Theme: Clean, Professional Blues
st.markdown("""
    <style>
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e1e4e8; }
    .hr-info-box { 
        background-color: #f8fafc; 
        color: #334155; 
        padding: 20px; 
        border-radius: 10px; 
        border-left: 5px solid #0f172a;
        line-height: 1.6;
    }
    .hr-info-box b { color: #0f172a; }
    </style>
    """, unsafe_allow_html=True)

# --- UTILITY FUNCTIONS ---
def extract_text_from_file(uploaded_file):
    try:
        if uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            return "".join([page.extract_text() or "" for page in pdf_reader.pages])
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(uploaded_file)
            return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

@st.cache_resource
def load_pipelines():
    # Model 1: The "Vibe" Matcher
    grader_pipe = pipeline("text-classification", model="Cheykong/HRVibeCheck")
    # Model 2: Key Info Extractor
    extractor_pipe = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    return grader_pipe, extractor_pipe

grader_pipe, extractor_pipe = load_pipelines()

def main():
    st.title("👔 HRVibeCheck: Smart HR Assistant")
    st.caption("Advanced Candidate Matching Tool | ISOM5240 L2")

    # --- HR-FRIENDLY METHODOLOGY ---
    with st.expander("🔍 Understanding the Vibe Check Analysis", expanded=True):
        st.markdown("""
        <div class="hr-info-box">
        <b>How we assess 'Fit':</b> Unlike traditional keyword tools that just look for specific words, 
        our AI uses <b>Contextual Intelligence</b>
