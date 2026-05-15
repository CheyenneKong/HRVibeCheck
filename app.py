import streamlit as st
from transformers import pipeline
import PyPDF2
from docx import Document
import plotly.express as px
import pandas as pd

st.set_page_config(page_title="HRVibeCheck", page_icon="👔", layout="wide")

# Modern Dark-Friendly Styling
st.markdown("""
    <style>
    .stMetric { background-color: #1e2937; padding: 20px; border-radius: 12px; }
    .big-number { font-size: 3.8rem !important; font-weight: bold; }
    .skill-pill { background-color: #3b82f6; color: white; padding: 8px 18px; 
                  border-radius: 25px; margin: 4px; display: inline-block; font-weight: 500; }
    .resume-box { background-color: #0f172a; color: #e2e8f0; padding: 25px; 
                  border-radius: 12px; line-height: 1.8; white-space: pre-wrap; }
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

    with st.expander("📘 What is Retention Probability?", expanded=True):
        st.markdown("""
        **Retention Probability** is our AI’s prediction of how likely a candidate will **stay long-term** and succeed in the role.  
        It was trained on real historical hiring decisions (Hire vs Reject).  
        **Higher score = Higher predicted retention & better overall fit.**
        """)

    # Sidebar
    with st.sidebar:
        st.header("Candidate Information")
        candidate_name = st.text_input("Candidate Name", "John Doe")
        
        st.subheader("Resume")
        uploaded_file = st.file_uploader("Upload PDF or Word", type=["pdf", "docx"])
        manual_text = st.text_area("Or paste resume text", height=180)

        st.divider()
        analyze_btn = st.button("🚀 Analyze Candidate", type="primary", use_container_width=True)

    if uploaded_file:
        resume_text = extract_text_from_file(uploaded_file)
    else:
        resume_text = manual_text

    if analyze_btn and resume_text:
        with st.spinner("Analyzing resume with AI..."):
            ret_result = retention_pipe(resume_text[:512])[0]
            score = ret_result['score']
            is_strong = score > 0.55

            # Improved skill detection
            skill_labels = ["Python", "Java", "SQL", "Machine Learning", "AWS", "Docker", 
                           "Kubernetes", "Leadership", "Project Management", "Data Analysis", 
                           "PyTorch", "Communication", "Excel", "Power BI"]
            skill_result = skill_pipe(resume_text[:1200], skill_labels, multi_label=True)
            
            skills_df = pd.DataFrame({
                "Skill": skill_result['labels'],
                "Confidence": skill_result['scores']
            }).sort_values("Confidence", ascending=False)

        st.success("✅ Analysis Complete!")

        col1, col2 = st.columns([1.1, 2])
        
        with col1:
            st.metric(
                label="**Retention Probability**",
                value=f"{score:.1%}",
                delta="Strong Hire Potential" if is_strong else "Further Review Recommended",
                delta_color="normal" if is_strong else "inverse"
            )

        with col2:
            st.subheader("🔑 Top Skills Detected")
            top_skills = skills_df[skills_df['Confidence'] > 0.35].head(12)
            if not top_skills.empty:
                cols = st.columns(4)
                for i, row in enumerate(top_skills.itertuples()):
                    cols[i % 4].markdown(f"<span class='skill-pill'>{row.Skill}</span>", unsafe_allow_html=True)
            else:
                st.info("No strong skills detected from our skill database.")

        st.divider()

        # Better Resume Preview
        st.subheader("📄 Resume Preview")
        st.markdown(f"""
        <div class="resume-box">
        {resume_text}
        </div>
        """, unsafe_allow_html=True)

    elif analyze_btn:
        st.warning("Please provide resume content.")

if __name__ == "__main__":
    main()
