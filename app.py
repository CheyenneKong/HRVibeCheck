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
    .hr-card { background-color: white; padding: 25px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading AI Models...")
def load_pipelines():
    retention_pipe = pipeline("text-classification", 
                             model="Cheykong/HRVibeCheck-Retention-Predictor", 
                             device=-1)
    skill_pipe = pipeline("zero-shot-classification", 
                         model="MoritzLaurer/deberta-v3-base-zeroshot-v2.0", 
                         device=-1)
    return retention_pipe, skill_pipe

retention_pipe, skill_pipe = load_pipelines()

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
    st.caption("AI-Powered Resume Screening • Retention + Skills Intelligence")

    with st.expander("📘 What is Retention Probability?", expanded=False):
        st.markdown("""
        **Retention Probability** is our AI’s prediction of how likely a candidate is to **stay long-term** and succeed in the role.  
        It is trained on real historical hiring decisions (Hire vs Reject).  
        Higher score = Higher predicted retention & better overall fit.
        """)

    # Sidebar
    with st.sidebar:
        st.header("Candidate Information")
        candidate_name = st.text_input("Candidate Name", "John Doe")
        
        st.subheader("Resume")
        uploaded_file = st.file_uploader("Upload PDF or Word", type=["pdf", "docx"])
        manual_text = st.text_area("Or paste resume text", height=200)

        st.divider()
        analyze_btn = st.button("🚀 Analyze Candidate", type="primary", use_container_width=True)

    if uploaded_file:
        resume_text = extract_text_from_file(uploaded_file)
    else:
        resume_text = manual_text

    if analyze_btn and resume_text:
        with st.spinner("Analyzing with AI..."):
            # Pipeline 1
            ret_result = retention_pipe(resume_text[:512])[0]
            score = ret_result['score']
            is_strong = score > 0.55

            # Pipeline 2
            skill_labels = ["Python", "SQL", "Machine Learning", "AWS", "Docker", "Kubernetes", 
                           "Leadership", "Project Management", "Data Analysis", "PyTorch", "Communication"]
            skill_result = skill_pipe(resume_text[:1000], skill_labels, multi_label=True)
            
            skills_df = pd.DataFrame({
                "Skill": skill_result['labels'],
                "Confidence": skill_result['scores']
            }).sort_values("Confidence", ascending=False)

        st.success("✅ Analysis Complete!")

        # ==================== MAIN RESULTS ====================
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric(
                label="**Retention Probability**",
                value=f"{score:.1%}",
                delta="Strong Hire Potential" if is_strong else "Further Review Recommended",
                delta_color="normal" if is_strong else "inverse"
            )

        with col2:
            st.subheader("🔑 Top Skills Detected")
            top_skills = skills_df[skills_df['Confidence'] > 0.35].head(8)
            if not top_skills.empty:
                fig = px.bar(top_skills, x="Skill", y="Confidence", 
                            text_auto='.1%', color="Confidence",
                            color_continuous_scale="Blues")
                fig.update_layout(height=380, xaxis_title="", yaxis_title="Confidence Score")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No strong skills matched.")

        st.divider()

        with st.expander("📄 Resume Preview
