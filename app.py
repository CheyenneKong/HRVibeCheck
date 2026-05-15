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
    .strength-pill { background-color: #10b981; color: white; padding: 8px 18px; border-radius: 25px; margin: 4px; display: inline-block; }
    .redflag-pill { background-color: #ef4444; color: white; padding: 8px 18px; border-radius: 25px; margin: 4px; display: inline-block; }
    .resume-box { background-color: #0f172a; color: #e2e8f0; padding: 25px; border-radius: 12px; line-height: 1.8; }
    </style>
    """, unsafe_allow_html=True)

# ==================== LOAD PIPELINES ====================
@st.cache_resource(show_spinner="Loading AI Models...")
def load_pipelines():
    pipe1 = pipeline("text-classification", 
                    model="Cheykong/HRVibeCheck-Retention-Predictor", 
                    device=-1)
    
    pipe2 = pipeline("zero-shot-classification", 
                    model="MoritzLaurer/deberta-v3-base-zeroshot-v2.0", 
                    device=-1)
    
    # Pipeline 3 uses the same reliable model
    pipe3 = pipeline("zero-shot-classification", 
                    model="MoritzLaurer/deberta-v3-base-zeroshot-v2.0", 
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
    st.caption("AI-Powered Resume Screening System")

    with st.expander("📘 What is Hire Recommendation Score?", expanded=True):
        st.markdown("""
        **Hire Recommendation Score** (0–100%) represents our AI’s confidence in recommending a candidate for hire.

        **How it is calculated**: Fine-tuned on real Hire vs Reject decisions.  
        **Score Interpretation**:
        | Score Range     | Recommendation       | Meaning |
        |-----------------|----------------------|--------|
        | **75% – 100%**  | **Strong Hire**      | Excellent fit |
        | **60% – 74%**   | **Good Hire**        | Solid candidate |
        | **45% – 59%**   | **Moderate Fit**     | Needs more evaluation |
        | **Below 45%**   | **Further Review**   | High risk |
        """)

    with st.sidebar:
        st.header("Candidate Information")
        candidate_name = st.text_input("Candidate Name", "John Doe")
        
        st.subheader("Resume")
        uploaded_file = st.file_uploader("Upload PDF or Word", type=["pdf", "docx"])
        manual_text = st.text_area("Or paste resume text", height=200)

        st.divider()
        analyze_btn = st.button("🚀 Run Full Analysis", type="primary", use_container_width=True)

    if uploaded_file:
        resume_text = extract_text_from_file(uploaded_file)
    else:
        resume_text = manual_text

    if analyze_btn and resume_text:
        with st.spinner("Analyzing resume with 3 pipelines..."):
            # Pipeline 1: Hire Score
            p1 = pipe1(resume_text[:512])[0]
            score = p1['score']
            is_strong = score > 0.55

            # Pipeline 2: Skills
            skill_labels = ["Python", "SQL", "Machine Learning", "AWS", "Docker", "Kubernetes", 
                           "Leadership", "Project Management", "Data Analysis", "PyTorch", "Communication"]
            p2 = pipe2(resume_text[:1000], skill_labels, multi_label=True)
            skills_df = pd.DataFrame({"Skill": p2['labels'], "Confidence": p2['scores']})
            top_skills = skills_df[skills_df['Confidence'] > 0.35].head(10)

            # Pipeline 3: Strengths & Red Flags
            strength_labels = ["Strong Leadership", "Technical Expertise", "Stable Career", 
                              "Fast Learner", "Results Driven", "Good Communication"]
            redflag_labels = ["Job Hopping", "Career Gaps", "Lack of Experience", 
                             "Irrelevant Background", "Weak Technical Skills"]

            p3_strength = pipe3(resume_text[:1200], strength_labels, multi_label=True)
            p3_redflag = pipe3(resume_text[:1200], redflag_labels, multi_label=True)

            top_strengths = [label for label, score in zip(p3_strength['labels'], p3_strength['scores']) if score > 0.4][:5]
            top_redflags = [label for label, score in zip(p3_redflag['labels'], p3_redflag['scores']) if score > 0.35][:4]

        st.success("✅ Full Analysis Complete!")

        col1, col2 = st.columns([1.2, 2])
        with col1:
            st.metric("**Hire Recommendation Score**", f"{score:.1%}",
                      delta="Strong Hire" if is_strong else "Further Review",
                      delta_color="normal" if is_strong else "inverse")

        with col2:
            st.subheader("🔑 Top Skills Detected")
            if not top_skills.empty:
                cols = st.columns(4)
                for i, row in enumerate(top_skills.itertuples()):
                    cols[i % 4].markdown(f"<span class='skill-pill'>{row.Skill}</span>", unsafe_allow_html=True)

        st.divider()

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("✅ Key Strengths")
            if top_strengths:
                for s in top_strengths:
                    st.markdown(f"<span class='strength-pill'>{s}</span>", unsafe_allow_html=True)
            else:
                st.write("No major strengths detected.")

        with c2:
            st.subheader("⚠️ Potential Red Flags")
            if top_redflags:
                for r in top_redflags:
                    st.markdown(f"<span class='redflag-pill'>{r}</span>", unsafe_allow_html=True)
            else:
                st.success("No major red flags detected.")

        st.divider()
        st.subheader("📄 Resume Preview")
        st.markdown(f"<div class='resume-box'>{resume_text}</div>", unsafe_allow_html=True)

    elif analyze_btn:
        st.warning("Please provide resume content.")

if __name__ == "__main__":
    main()
