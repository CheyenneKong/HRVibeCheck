import streamlit as st
from transformers import pipeline
import PyPDF2
from docx import Document

# --- PAGE CONFIG ---
st.set_page_config(page_title="HRVibeCheck", page_icon="👔", layout="wide")

st.markdown("""
    <style>
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e1e4e8; }
    .hr-info-box { background-color: #f8fafc; color: #334155; padding: 20px; border-radius: 10px; border-left: 5px solid #0f172a; }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD PIPELINES ---
@st.cache_resource
def load_pipelines():
    # Pipeline 1: Retention Prediction (Your fine-tuned model)
    retention_pipe = pipeline(
        "text-classification", 
        model="Cheykong/HRVibeCheck-Retention-Predictor"
    )
    
    # Pipeline 2: Skill Extraction (Zero-shot)
    skill_pipe = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=0 if st.runtime.exists("cuda") else -1
    )
    
    return retention_pipe, skill_pipe

retention_pipe, skill_pipe = load_pipelines()

# --- HELPER FUNCTIONS ---
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
    st.caption("AI-Powered Resume Screening | Retention + Skills Analysis")

    with st.expander("🔍 How It Works", expanded=True):
        st.markdown("""
        **Pipeline 1**: Predicts **Retention Probability** (Hire vs Reject)  
        **Pipeline 2**: Extracts **Key Skills** from the resume
        """)

    # Sidebar
    st.sidebar.header("Candidate Information")
    candidate_name = st.sidebar.text_input("Candidate Name", "John Doe")

    st.sidebar.subheader("Resume")
    uploaded_file = st.sidebar.file_uploader("Upload Resume (PDF/Word)", type=["pdf", "docx"])
    manual_text = st.sidebar.text_area("Or paste resume text", height=200)

    if uploaded_file:
        resume_text = extract_text_from_file(uploaded_file)
    else:
        resume_text = manual_text

    analyze_button = st.sidebar.button("🚀 Analyze Candidate", type="primary")

    if analyze_button and resume_text:
        with st.spinner("Analyzing with both pipelines..."):
            # Pipeline 1: Retention
            retention_result = retention_pipe(resume_text[:512])[0]
            
            # Pipeline 2: Skills
            candidate_labels = ["Python", "Java", "SQL", "Machine Learning", "Deep Learning", 
                              "AWS", "Azure", "Docker", "Kubernetes", "Leadership", 
                              "Project Management", "Communication", "Data Analysis", "PyTorch"]
            
            skill_result = skill_pipe(resume_text[:1000], candidate_labels, multi_label=True)

        # Display Results
        st.subheader(f"Results for: {candidate_name}")

        col1, col2 = st.columns(2)
        with col1:
            score = retention_result['score']
            label = retention_result['label']
            is_good = score > 0.5
            st.metric(
                label="Retention / Hire Probability",
                value=f"{score:.1%}",
                delta="Strong Hire" if is_good else "Further Review",
                delta_color="normal" if is_good else "inverse"
            )

        with col2:
            st.write("**Extracted Key Skills**")
            skills = [label for label, score in zip(skill_result['labels'], skill_result['scores']) if score > 0.4]
            for skill in skills[:10]:
                st.success(f"• {skill}")

        st.divider()
        st.write("**Resume Preview**")
        st.info(resume_text[:800] + "..." if len(resume_text) > 800 else resume_text)

    elif analyze_button:
        st.warning("Please provide resume text or upload a file.")

if __name__ == "__main__":
    main()
