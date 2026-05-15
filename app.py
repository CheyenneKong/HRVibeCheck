import streamlit as st
from transformers import pipeline
import PyPDF2
from docx import Document

# --- PAGE CONFIG ---
st.set_page_config(page_title="HRVibeCheck", page_icon="👔", layout="wide")

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
        return None
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

@st.cache_resource
def load_pipelines():
    # Pipeline 1: Updated to your latest fine-tuned model
    grader_pipe = pipeline(
        "text-classification", 
        model="Cheykong/HRVibeCheck-Retention-Predictor"
    )
    
    # Pipeline 2: Skill Extraction (NER)
    extractor_pipe = pipeline(
        "ner", 
        model="dslim/bert-base-NER", 
        aggregation_strategy="simple"
    )
    return grader_pipe, extractor_pipe

grader_pipe, extractor_pipe = load_pipelines()

def main():
    st.title("👔 HRVibeCheck: Smart HR Assistant")
    st.caption("Retention Prediction + Skills Extraction | ISOM5240 Group Project")

    with st.expander("🔍 How HRVibeCheck Works", expanded=True):
        st.markdown("""
        <div class="hr-info-box">
        <b>Two-Pipeline AI System:</b><br>
        • <b>Pipeline 1 (Retention Predictor)</b>: Analyzes the resume to predict if the candidate is likely to stay long-term.<br>
        • <b>Pipeline 2 (Skills Extractor)</b>: Automatically identifies key skills, technologies, and experiences.
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # --- SIDEBAR ---
    st.sidebar.header("📥 Candidate Input")
    candidate_name = st.sidebar.text_input("Candidate Name", "John
