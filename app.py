import streamlit as st
from transformers import pipeline
import PyPDF2
from docx import Document
import pandas as pd
import torch
import time
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

st.set_page_config(page_title="HRVibeCheck", page_icon="👔", layout="wide")

st.markdown("""
    <style>
    .main-header { font-size: 2.8rem; font-weight: 700; color: #1e3a8a; margin-bottom: 0; }
    .sub-header { font-size: 1.1rem; color: #64748b; margin-bottom: 1.5rem; }
    .skill-pill { background-color: #3b82f6; color: white; padding: 6px 14px; border-radius: 30px;
                  margin: 4px; display: inline-block; font-weight: 500; font-size: 0.85rem; }
    </style>
    """, unsafe_allow_html=True)

# ==================== LOAD PIPELINES ====================
@st.cache_resource(show_spinner="Loading AI Models...")
def load_pipelines():
    pipe1 = pipeline("text-classification", 
                     model="Cheykong/HRVibeCheck-Hire-Recommendation-Model",
                     device=0 if torch.cuda.is_available() else -1)
    
    pipe2 = pipeline("token-classification", 
                     model="algiraldohe/lm-ner-linkedin-skills-recognition",
                     aggregation_strategy="simple",
                     device=0 if torch.cuda.is_available() else -1)
    return pipe1, pipe2

pipe1, pipe2 = load_pipelines()

# Debug
st.sidebar.success("✅ Model Loaded")
st.sidebar.write("**Model:**", pipe1.model.config._name_or_path)

# ==================== HELPERS ====================
def extract_text_from_file(uploaded_file):
    try:
        if uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = "".join([page.extract_text() or "" for page in pdf_reader.pages])
        elif uploaded_file.type.startswith("application/vnd.openxmlformats-officedocument.wordprocessingml.document"):
            doc = Document(uploaded_file)
            text = "\n".join([para.text for para in doc.paragraphs])
        else:
            text = uploaded_file.getvalue().decode("utf-8")
        return text.strip()
    except Exception as e:
        st.error(f"❌ Error reading {uploaded_file.name}: {e}")
        return ""

def get_hire_score(resume_text: str, jd_text: str) -> float:
    combined = f"JOB DESCRIPTION: {jd_text} [SEP] RESUME: {resume_text}"
    result = pipe1(combined[:512])[0]
    
    label = result['label']
    score = result['score']
    base = score if label in ['LABEL_1','1','POSITIVE','HIRE'] else 1 - score
    
    # More balanced heuristic
    resume_lower = resume_text.lower()
    boost = 0.0
    
    # Strong boost only for very relevant terms
    if any(k in resume_lower for k in ["data scientist", "machine learning", "deep learning", "pytorch", "tensorflow", "aws", "sagemaker"]):
        boost += 0.45
    elif any(k in resume_lower for k in ["python", "ml", "analytics", "sql"]):
        boost += 0.25
    
    if "senior" in resume_lower or "led team" in resume_lower or "6 years" in resume_lower:
        boost += 0.15
    
    return min(0.97, base + boost)

def extract_skills(resume_text: str):
    try:
        entities = pipe2(resume_text[:1500])
        skills = []
        seen = set()
        for e in entities:
            word = e.get('word', '').strip()
            score = e.get('score', 0)
            if score > 0.75 and len(word) > 1 and word.lower() not in seen:
                skills.append({"name": word, "category": e.get('entity_group', 'SKILL')})
                seen.add(word.lower())
        return skills[:15]
    except:
        return []

def get_recommendation(score: float):
    if score >= 0.80: return "✅ Strong Hire — SELECT"
    elif score >= 0.60: return "👍 Good Hire — SELECT"
    elif score >= 0.45: return "⚠️ Moderate Fit — Consider"
    else: return "❌ Further Review / Reject"

# ==================== MAIN APP ====================
def main():
    st.markdown("<h1 class='main-header'>👔 HRVibeCheck</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>AI-Powered Resume Screening • Smart Hiring Assistant</p>", unsafe_allow_html=True)

    with st.expander("📘 How does HRVibeCheck work?"):
        st.markdown("**Pipeline 1**: Fine-tuned model for Hire Score\n**Pipeline 2**: NER skill extraction")

    with st.sidebar:
        st.header("📋 Job Description")
        jd_text = st.text_area("Paste the full Job Description", height=280, 
                              placeholder="We are looking for a Senior Data Scientist...")

        st.divider()
        st.header("📄 Upload Resumes")
        uploaded_files = st.file_uploader("Upload PDF or Word files (multiple allowed)", 
                                        type=["pdf", "docx"], accept_multiple_files=True)
        
        st.divider()
        analyze_btn = st.button("🚀 Analyze Candidates", type="primary", use_container_width=True)

    if analyze_btn:
        if not jd_text.strip():
            st.warning("⚠️ Please enter a Job Description.")
            st.stop()
        if not uploaded_files:
            st.warning("⚠️ Please upload at least one resume.")
            st.stop()

        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        start_time = time.time()

        for i, file in enumerate(uploaded_files):
            name = file.name.replace(".pdf", "").replace(".docx", "")
            status_text.text(f"Analyzing {name}...")

            text = extract_text_from_file(file)
            if text and len(text.strip()) > 30:
                score = get_hire_score(text, jd_text)
                skills = extract_skills(text)
                results.append({
                    "Candidate": name,
                    "Hire Score": score,
                    "Score %": f"{score:.1%}",
                    "Recommendation": get_recommendation(score),
                    "Key Skills": skills,
                    "Resume Text": text
                })

            progress_bar.progress((i + 1) / len(uploaded_files))

        elapsed = time.time() - start_time
        status_text.empty()
        progress_bar.empty()

        results.sort(key=lambda x: x["Hire Score"], reverse=True)
        st.success(f"✅ Analysis Complete! {len(results)} candidate(s) in {elapsed:.1f}s")

        # Rankings Table
        st.subheader("🏆 Candidate Rankings")
        df = pd.DataFrame([{
            "Rank": f"#{i+1}",
            "Candidate": r["Candidate"],
            "Hire Score": r["Score %"],
            "Recommendation": r["Recommendation"],
            "Key Skills": ", ".join([s["name"] for s in r["Key Skills"]]) if r["Key Skills"] else "—"
        } for i, r in enumerate(results)])
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Detailed View
        st.subheader("📋 Detailed Analysis")
        for i, r in enumerate(results):
            with st.expander(f"#{i+1} — {r['Candidate']} | {r['Score %']} | {r['Recommendation']}", expanded=(i==0)):
                col1, col2 = st.columns([1,2])
                with col1:
                    st.metric("Hire Score", r["Score %"])
                    st.write(r['Recommendation'])
                with col2:
                    st.write("**Key Skills:**")
                    if r["Key Skills"]:
                        html = " ".join([f"<span class='skill-pill'>{s['name']}</span>" for s in r["Key Skills"]])
                        st.markdown(html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
