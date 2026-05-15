import streamlit as st
from transformers import pipeline
import PyPDF2
from docx import Document

st.set_page_config(page_title="HRVibeCheck", page_icon="👔", layout="wide")

# --- LIGHTER PIPELINES ---
@st.cache_resource
def load_pipelines():
    # Pipeline 1: Your fine-tuned model
    retention_pipe = pipeline(
        "text-classification", 
        model="Cheykong/HRVibeCheck-Retention-Predictor",
        device=-1   # Force CPU to avoid memory issues
    )
    
    # Pipeline 2: Lighter zero-shot model
    skill_pipe = pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/deberta-v3-base-zeroshot-v2.0",   # Much lighter & faster
        device=-1
    )
    return retention_pipe, skill_pipe

retention_pipe, skill_pipe = load_pipelines()

# --- Rest of your functions (extract_text_from_file) ---
def extract_text_from_file(uploaded_file):
    try:
        if uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            return "".join([page.extract_text() or "" for page in pdf_reader.pages])
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(uploaded_file)
            return "\n".join([para.text for para in doc.paragraphs])
        return None
    except:
        return None

# --- MAIN APP ---
def main():
    st.title("👔 HRVibeCheck")
    st.caption("Retention Prediction + Skills Extraction")

    st.sidebar.header("Candidate Input")
    candidate_name = st.sidebar.text_input("Candidate Name", "John Doe")
    
    uploaded_file = st.sidebar.file_uploader("Upload Resume (PDF/Word)", type=["pdf", "docx"])
    manual_text = st.sidebar.text_area("Or paste resume text", height=200)

    if uploaded_file:
        resume_text = extract_text_from_file(uploaded_file)
    else:
        resume_text = manual_text

    if st.sidebar.button("🚀 Analyze Candidate", type="primary") and resume_text:
        with st.spinner("Analyzing..."):
            # Pipeline 1
            retention_result = retention_pipe(resume_text[:512])[0]
            
            # Pipeline 2 - Lighter labels
            labels = ["Python", "Java", "SQL", "Machine Learning", "AWS", "Docker", 
                     "Leadership", "Project Management", "Data Analysis", "PyTorch"]
            
            skill_result = skill_pipe(resume_text[:800], labels, multi_label=True)

        # Display
        col1, col2 = st.columns(2)
        with col1:
            score = retention_result['score']
            st.metric("Retention Probability", f"{score:.1%}")
        
        with col2:
            st.write("**Key Skills Detected**")
            skills = [label for label, score in zip(skill_result['labels'], skill_result['scores']) if score > 0.35]
            for skill in skills:
                st.success(f"• {skill}")

        st.info(resume_text[:700] + "..." if len(resume_text) > 700 else resume_text)

if __name__ == "__main__":
    main()
