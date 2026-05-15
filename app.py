import streamlit as st
from transformers import pipeline
import PyPDF2
from docx import Document
import pandas as pd

st.set_page_config(page_title="HRVibeCheck", page_icon="👔", layout="wide")

st.markdown("""
    <style>
    .main-header { font-size: 2.8rem; font-weight: 700; color: #1e3a8a; margin-bottom: 0; }
    .sub-header { font-size: 1.1rem; color: #64748b; }
    .stMetric { background-color: white; padding: 25px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
    .skill-pill { background-color: #3b82f6; color: white; padding: 10px 20px; border-radius: 30px; margin: 5px; display: inline-block; font-weight: 500; }
    .seniority-box { background: linear-gradient(135deg, #6366f1, #4f46e5); color: white; padding: 20px; border-radius: 16px; font-size: 1.15em; font-weight: 600; }
    .resume-box { background-color: #0f172a; color: #e2e8f0; padding: 28px; border-radius: 16px; line-height: 1.85; border-left: 5px solid #64748b; }
    </style>
    """, unsafe_allow_html=True)

# ==================== LOAD PIPELINES ====================
@st.cache_resource(show_spinner="Loading AI Models...")
def load_pipelines():
    pipe1 = pipeline("text-classification", 
                    model="Cheykong/HRVibeCheck-Retention-Predictor", 
                    device=-1)
    
    pipe23 = pipeline("zero-shot-classification", 
                     model="MoritzLaurer/deberta-v3-base-zeroshot-v2.0", 
                     device=-1)
    
    return pipe1, pipe23

pipe1, pipe23 = load_pipelines()

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
    st.markdown("<h1 class='main-header'>👔 HRVibeCheck</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>AI-Powered Resume Screening • Smart Hiring Assistant</p>", unsafe_allow_html=True)

    with st.expander("📘 What is Hire Recommendation Score?", expanded=False):
        st.markdown("""
        **Hire Recommendation Score** (0–100%) represents our AI’s confidence in recommending a candidate for hire.

        **How it is calculated**: The model was fine-tuned on real historical hiring decisions (`Hire` vs `Reject`).  
        **Score Interpretation**:
        | Score Range     | Recommendation       | Meaning |
        |-----------------|----------------------|--------|
        | **75% – 100%**  | **Strong Hire**      | Excellent fit |
        | **60% – 74%**   | **Good Hire**        | Solid candidate |
        | **45% – 59%**   | **Moderate Fit**     | Needs more evaluation |
        | **Below 45%**   | **Further Review**   | High risk |
        """)

    with st.sidebar:
        st.header("Resume Input")
        
        uploaded_file = st.file_uploader("Upload PDF or Word File", type=["pdf", "docx"])
        manual_text = st.text_area("Or paste resume text here", height=250, 
                                  placeholder="Paste candidate resume text...")

        st.divider()
        analyze_btn = st.button("🚀 Run Full Analysis", type="primary", use_container_width=True)

    if uploaded_file:
        resume_text = extract_text_from_file(uploaded_file)
    else:
        resume_text = manual_text

    if analyze_btn and resume_text:
        with st.spinner("🤖 Analyzing resume with 3 AI pipelines..."):
            # Pipeline 1: Hire Recommendation
            p1 = pipe1(resume_text[:512])[0]
            score = p1['score']
            is_strong = score > 0.55

            # Pipeline 2: Top Skills
            skill_labels = ["Python", "SQL", "Machine Learning", "AWS", "Docker", "Kubernetes",
                           "Leadership", "Project Management", "Data Analysis", "PyTorch", "Communication"]
            p2 = pipe23(resume_text[:1000], skill_labels, multi_label=True)
            top_skills = [label for label, sc in zip(p2['labels'], p2['scores']) if sc > 0.35][:8]

            # Pipeline 3: Seniority
            seniority_labels = ["Senior Level (5+ years)", "Mid Level (2-5 years)",
                               "Junior Level (0-2 years)", "Entry Level / Fresh Graduate"]
            p3 = pipe23(resume_text[:1500], seniority_labels, multi_label=False)
            predicted_level = p3['labels'][0]
            confidence = p3['scores'][0]

        st.success("✅ Analysis Complete!")

        tab1, tab2, tab3 = st.tabs(["📊 Assessment", "🔑 Skills & Seniority", "📄 Resume"])

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("**Hire Recommendation Score**", f"{score:.1%}",
                          delta="Strong Hire" if is_strong else "Further Review",
                          delta_color="normal" if is_strong else "inverse")

        with tab2:
            st.subheader("🔑 Top Skills Detected")
            if top_skills:
                cols = st.columns(4)
                for i, skill in enumerate(top_skills):
                    cols[i % 4].markdown(f"<span class='skill-pill'>{skill}</span>", unsafe_allow_html=True)
            else:
                st.info("No strong skills detected.")

            st.divider()
            st.subheader("📊 Candidate Seniority / Experience Level")
            st.markdown(f"""
            <div class='seniority-box'>
                Predicted Level: {predicted_level}<br>
                Confidence: {confidence:.1%}
            </div>
            """, unsafe_allow_html=True)

        with tab3:
            st.subheader("📄 Original Resume")
            st.markdown(f"<div class='resume-box'>{resume_text}</div>", unsafe_allow_html=True)

    elif analyze_btn:
        st.warning("Please upload a file or paste resume text.")

if __name__ == "__main__":
    main()
