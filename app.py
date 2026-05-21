import streamlit as st
from transformers import pipeline
import PyPDF2
from docx import Document
import pandas as pd
import torch
import time
import os

# Suppress tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

st.set_page_config(
    page_title="HRVibeCheck",
    page_icon="👔",
    layout="wide"
)
st.markdown("""
    <style>
    .main-header { font-size: 2.8rem; font-weight: 700; color: #1e3a8a; margin-bottom: 0; }
    .sub-header { font-size: 1.1rem; color: #64748b; margin-bottom: 1.5rem; }
    .skill-pill { background-color: #3b82f6; color: white; padding: 6px 14px; border-radius: 30px;
                  margin: 4px; display: inline-block; font-weight: 500; font-size: 0.85rem; }
    .rank-badge { background: linear-gradient(135deg, #1e3a8a, #3b82f6); color: white;
                  padding: 4px 12px; border-radius: 20px; font-weight: 700; }
    .score-box { background-color: white; padding: 20px; border-radius: 16px;
                 box-shadow: 0 4px 12px rgba(0,0,0,0.08); text-align: center; }
    </style>
    """, unsafe_allow_html=True)

# ==================== LOAD PIPELINES ====================
@st.cache_resource(show_spinner="Loading AI Models... This may take 20-40 seconds.")
def load_pipelines():
    # Pipeline 1: Fine-tuned Hire Recommendation Model
    pipe1 = pipeline(
        "text-classification",
        model="Cheykong/HRVibeCheck-Hire-Recommendation-Model",   # ← Change only if your HF username/repo is different
        device=0 if torch.cuda.is_available() else -1
    )
    
    # Pipeline 2: Skill Extraction (Best performing model)
    pipe2 = pipeline(
        "token-classification",
        model="algiraldohe/lm-ner-linkedin-skills-recognition",
        aggregation_strategy="simple",
        device=0 if torch.cuda.is_available() else -1
    )
    return pipe1, pipe2

pipe1, pipe2 = load_pipelines()

# ==================== HELPER FUNCTIONS ====================
def extract_text_from_file(uploaded_file):
    """Extract text from PDF or DOCX with better error handling."""
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
    """Pipeline 1: Return hire probability (0-1)."""
    combined = f"JOB DESCRIPTION: {jd_text} [SEP] RESUME: {resume_text}"
    result = pipe1(combined[:512])[0]
    
    label = result['label']
    score = result['score']
    
    # Handle common label formats from fine-tuned models
    if label in ['LABEL_1', '1', 'POSITIVE', 'HIRE', 'hire']:
        return score
    else:
        return 1 - score

def extract_skills(resume_text: str):
    """Pipeline 2: Extract high-confidence skills."""
    try:
        entities = pipe2(resume_text[:1500])
        skills = []
        seen = set()
        
        for e in entities:
            word = e['word'].strip()
            score = e.get('score', 0)
            label = e.get('entity_group', 'SKILL')
            
            if score > 0.75 and len(word) > 1 and word.lower() not in seen:
                skills.append({"name": word, "category": label})
                seen.add(word.lower())
        
        return skills[:15]  # limit to top 15
    except:
        return []

def get_recommendation(score: float):
    if score >= 0.75:
        return "✅ Strong Hire — SELECT"
    elif score >= 0.60:
        return "👍 Good Hire — SELECT"
    elif score >= 0.45:
        return "⚠️ Moderate Fit — Consider"
    else:
        return "❌ Further Review / Reject"

# ==================== MAIN APP ====================
def main():
    st.markdown("<h1 class='main-header'>👔 HRVibeCheck</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>AI-Powered Resume Screening • Smart Hiring Assistant</p>", unsafe_allow_html=True)

    with st.expander("📘 How does HRVibeCheck work?", expanded=False):
        st.markdown("""
        **HRVibeCheck** uses two deep learning pipelines:
        - **Pipeline 1**: Fine-tuned transformer that compares Job Description vs Resume and gives a **Hire Score**.
        - **Pipeline 2**: NER model that automatically extracts key skills from the resume.
        
        This helps recruiters screen candidates faster and more objectively.
        """)

    # Sidebar
    with st.sidebar:
        st.header("📋 Job Description")
        jd_text = st.text_area(
            "Paste the full Job Description",
            height=250,
            placeholder="We are looking for a Senior Data Scientist with strong Python, Machine Learning..."
        )
        
        st.divider()
        st.header("📄 Upload Resumes")
        uploaded_files = st.file_uploader(
            "Upload PDF or Word files (multiple allowed)",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            help="You can upload several candidate resumes at once"
        )
        
        st.divider()
        analyze_btn = st.button("🚀 Analyze Candidates", type="primary", use_container_width=True)

    # Analysis
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

        for i, uploaded_file in enumerate(uploaded_files):
            candidate_name = uploaded_file.name.replace(".pdf", "").replace(".docx", "")
            status_text.text(f"🔍 Analyzing {candidate_name}... ({i+1}/{len(uploaded_files)})")

            resume_text = extract_text_from_file(uploaded_file)
            
            if resume_text and len(resume_text.strip()) > 50:
                hire_score = get_hire_score(resume_text, jd_text)
                skills = extract_skills(resume_text)
                
                results.append({
                    "Candidate": candidate_name,
                    "Hire Score": hire_score,
                    "Score %": f"{hire_score:.1%}",
                    "Recommendation": get_recommendation(hire_score),
                    "Key Skills": skills,
                    "Resume Text": resume_text
                })
            else:
                st.warning(f"⚠️ Could not extract meaningful text from {uploaded_file.name}")

            progress_bar.progress((i + 1) / len(uploaded_files))

        # Finalize
        elapsed = time.time() - start_time
        status_text.empty()
        progress_bar.empty()

        if not results:
            st.error("No resumes could be processed.")
            st.stop()

        results.sort(key=lambda x: x["Hire Score"], reverse=True)
        st.success(f"✅ Analysis Complete! {len(results)} candidate(s) processed in {elapsed:.1f} seconds.")

        # Rankings Table
        st.subheader("🏆 Candidate Rankings")
        table_data = [{
            "Rank": f"#{rank}",
            "Candidate": r["Candidate"],
            "Hire Score": r["Score %"],
            "Recommendation": r["Recommendation"],
            "Key Skills": ", ".join([s["name"] for s in r["Key Skills"]]) if r["Key Skills"] else "—"
        } for rank, r in enumerate(results, 1)]

        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

        # Detailed Cards
        st.subheader("📋 Detailed Analysis")
        for rank, r in enumerate(results, 1):
            with st.expander(f"#{rank} — {r['Candidate']} | {r['Score %']} | {r['Recommendation']}", expanded=(rank == 1)):
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric("Hire Score", r["Score %"])
                    st.write(f"**Recommendation:** {r['Recommendation']}")
                with col2:
                    st.write("**Extracted Key Skills:**")
                    if r["Key Skills"]:
                        skill_html = " ".join([
                            f"<span class='skill-pill' title='{s['category']}'>{s['name']}</span>"
                            for s in r["Key Skills"]
                        ])
                        st.markdown(skill_html, unsafe_allow_html=True)
                    else:
                        st.info("No high-confidence skills detected.")

                st.divider()
                st.write("**Resume Preview:**")
                preview = r["Resume Text"][:700] + "..." if len(r["Resume Text"]) > 700 else r["Resume Text"]
                st.text_area("", preview, height=200, disabled=True)

if __name__ == "__main__":
    main()
