import streamlit as st
from transformers import pipeline
import PyPDF2
from docx import Document
import plotly.express as px
import pandas as pd

st.set_page_config(page_title="HRVibeCheck", page_icon="👔", layout="wide")

st.markdown("""
    <style>
    .stMetric { background-color: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
    .skill-pill { background-color: #3b82f6; color: white; padding: 8px 18px; border-radius: 25px; margin: 4px; display: inline-block; }
    .resume-box { background-color: #0f172a; color: #e2e8f0; padding: 25px; border-radius: 12px; line-height: 1.8; }
    </style>
    """, unsafe_allow_html=True)

# ==================== LOAD ALL 3 PIPELINES ====================
@st.cache_resource(show_spinner="Loading AI Models...")
def load_pipelines():
    # Pipeline 1: Hire Recommendation (Fine-tuned)
    pipe1 = pipeline("text-classification", 
                    model="Cheykong/HRVibeCheck-Retention-Predictor", 
                    device=-1)
    
    # Pipeline 2: Skill Extraction
    pipe2 = pipeline("zero-shot-classification", 
                    model="MoritzLaurer/deberta-v3-base-zeroshot-v2.0", 
                    device=-1)
    
    # Pipeline 3: Resume Summarization
    pipe3 = pipeline("summarization", 
                    model="facebook/bart-large-cnn", 
                    device=-1)
    
    return pipe1, pipe2, pipe3

pipe1, pipe2, pipe3 = load_pipelines()

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

def main():
    st.title("👔 HRVibeCheck")
    st.caption("3-Pipeline AI Resume Screening System")

    with st.expander("📘 About Our AI System", expanded=True):
        st.markdown("""
        - **Pipeline 1**: Hire Recommendation Score (Fine-tuned Model)  
        - **Pipeline 2**: Automatic Skill Extraction  
        - **Pipeline 3**: Professional Resume Summarization
        """)

    # Sidebar
    with st.sidebar:
        st.header("Candidate Information")
        candidate_name = st.text_input("Candidate Name", "John Doe")
        
        st.subheader("Resume")
        uploaded_file = st.file_uploader("Upload PDF or Word", type=["pdf", "docx"])
        manual_text = st.text_area("Or paste resume text", height=180)

        st.divider()
        analyze_btn = st.button("🚀 Run Full Analysis", type="primary", use_container_width=True)

    if uploaded_file:
        resume_text = extract_text_from_file(uploaded_file)
    else:
        resume_text = manual_text

    if analyze_btn and resume_text:
        with st.spinner("Running all 3 AI pipelines..."):
            # Pipeline 1
            p1_result = pipe1(resume_text[:512])[0]
            score = p1_result['score']
            is_strong = score > 0.55

            # Pipeline 2 - Skills
            skill_labels = ["Python", "SQL", "Machine Learning", "AWS", "Docker", "Kubernetes", 
                           "Leadership", "Project Management", "Data Analysis", "PyTorch"]
            p2_result = pipe2(resume_text[:1000], skill_labels, multi_label=True)
            skills_df = pd.DataFrame({"Skill": p2_result['labels'], "Confidence": p2_result['scores']})
            top_skills = skills_df[skills_df['Confidence'] > 0.35].head(10)

            # Pipeline 3 - Summarization
            summary = pipe3(resume_text[:2000], max_length=180, min_length=60, do_sample=False)[0]['summary_text']

        st.success("✅ Full Analysis Complete!")

        # Results Layout
        col1, col2 = st.columns([1.2, 2])
        with col1:
            st.metric(
                label="**Hire Recommendation Score**",
                value=f"{score:.1%}",
                delta="Strong Hire" if is_strong else "Further Review",
                delta_color="normal" if is_strong else "inverse"
            )

        with col2:
            st.subheader("🔑 Top Skills Detected")
            if not top_skills.empty:
                cols = st.columns(4)
                for i, row in enumerate(top_skills.itertuples()):
                    cols[i % 4].markdown(f"<span class='skill-pill'>{row.Skill}</span>", unsafe_allow_html=True)

        st.divider()

        # Pipeline 3 Result
        st.subheader("📝 AI-Generated Professional Summary")
        st.info(summary)

        st.divider()
        st.subheader("📄 Original Resume")
        st.markdown(f"<div class='resume-box'>{resume_text}</div>", unsafe_allow_html=True)

    elif analyze_btn:
        st.warning("Please provide resume content.")

if __name__ == "__main__":
    main()
