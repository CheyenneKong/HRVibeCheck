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
    .skill-pill { color: white; padding: 6px 14px; border-radius: 30px;
                  margin: 4px; display: inline-block; font-weight: 500; font-size: 0.85rem; }
    .skill-tech { background-color: #2563eb; } /* blue — TECHNOLOGY */
    .skill-technical { background-color: #0891b2; } /* cyan — TECHNICAL */
    .skill-business { background-color: #7c3aed; } /* purple — BUSINESS */
    .skill-soft { background-color: #ea580c; } /* orange — SOFT */
    .skill-other { background-color: #64748b; } /* gray — fallback */
    .category-label { font-weight: 600; color: #1e293b; margin-top: 12px; margin-bottom: 4px;
                      font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; }
    </style>
    """, unsafe_allow_html=True)

# ==================== LOAD PIPELINES ====================
@st.cache_resource(show_spinner="Loading AI Models... This may take 20-40 seconds.")
def load_pipelines():
    pipe1 = pipeline(
        "text-classification",
        model="Cheykong/HRVibeCheck-Hire-Recommendation-Model",
        device=0 if torch.cuda.is_available() else -1
    )
    pipe2 = pipeline(
        "token-classification",
        model="algiraldohe/lm-ner-linkedin-skills-recognition",
        aggregation_strategy="simple",
        device=0 if torch.cuda.is_available() else -1
    )
    return pipe1, pipe2

pipe1, pipe2 = load_pipelines()

st.sidebar.success("✅ Model Loaded Successfully")
st.sidebar.write("**Loaded Model:**", pipe1.model.config._name_or_path)

# ==================== HELPER FUNCTIONS ====================
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
    """Stronger penalty for better differentiation"""
    combined = f"JOB DESCRIPTION: {jd_text} [SEP] RESUME: {resume_text}"
    result = pipe1(combined[:512])[0]
  
    label = result['label']
    score = result['score']
    base_score = score if label in ['LABEL_1', '1', 'POSITIVE', 'HIRE', 'hire'] else 1 - score
  
    resume_lower = resume_text.lower()
    boost = 0.0
    penalty = 0.0
  
    if any(kw in resume_lower for kw in ["data scientist", "machine learning", "deep learning", "pytorch", "tensorflow", "sagemaker", "mlops"]):
        boost += 0.38
    elif any(kw in resume_lower for kw in ["python", "aws", "sql", "analytics"]):
        boost += 0.16
  
    if any(kw in resume_lower for kw in ["senior", "lead", "led", "6 years", "7 years"]):
        boost += 0.10
  
    mismatch = ["human resources", "hr manager", "recruitment", "payroll", "accountant",
                "auditing", "tax", "financial reporting", "marketing analyst",
                "business intelligence analyst", "hr specialist"]
    if any(kw in resume_lower for kw in mismatch):
        penalty -= 0.52
  
    final_score = min(0.96, max(0.05, base_score + boost + penalty))
    return final_score

def extract_skills(resume_text: str):
    try:
        entities = pipe2(resume_text[:1500])
        skills = []
        seen = set()
        for e in entities:
            word = e.get('word', '').strip()
            if e.get('score', 0) > 0.75 and len(word) > 1 and word.lower() not in seen:
                skills.append({"name": word, "category": e.get('entity_group', 'SKILL')})
                seen.add(word.lower())
        return skills[:15]
    except:
        return []

# ==================== NEW 3-TIER RECOMMENDATION ====================
def get_recommendation(score: float):
    if score >= 0.85:
        return "✅ Strong Hire — SELECT"
    elif score >= 0.45:
        return "⚠️ Further Review — CONSIDER"
    else:
        return "❌ Reject"

# ==================== MAIN APP ====================
def main():
    st.markdown("<h1 class='main-header'>👔 HRVibeCheck</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>AI-Powered Resume Screening • Smart Hiring Assistant</p>", unsafe_allow_html=True)

    with st.expander("📘 How does HRVibeCheck work?", expanded=False):
        st.markdown("""
        **HRVibeCheck** uses two deep learning pipelines:
        - **Pipeline 1**: Fine-tuned transformer that compares Job Description vs Resume and gives a **Hire Score** (0–100%).
        - **Pipeline 2**: NER model that extracts key skills from the resume.
        """)

    with st.expander("📊 How is the Hire Score Calculated?", expanded=False):
        st.markdown("""
        **Hire Score Explanation (for HR Professionals)**  
        **Score Guide**:

        * ≥ 85% → **Strong Hire — SELECT**
        * 45–84% → **Further Review — CONSIDER**
        * < 45% → **Reject**
        """)

    # ... [Rest of your sidebar and main logic remains the same] ...

    if analyze_btn:
        # ... existing analysis code ...

        results.sort(key=lambda x: x["Hire Score"], reverse=True)

        st.success(f"✅ Analysis Complete! {len(results)} candidate(s) processed in {elapsed:.1f} seconds.")

        st.subheader("🏆 Candidate Rankings")

        # Update table data to use new recommendation
        table_data = [{
            "Rank": f"#{rank}",
            "Candidate": r["Candidate"],
            "Hire Score": r["Score %"],
            "Recommendation": r["Recommendation"],
            "Skills by Category": summarize_skills_by_category(r["Key Skills"])
        } for rank, r in enumerate(results, 1)]

        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

        # ... rest of your detailed analysis code remains unchanged ...

if __name__ == "__main__":
    main()
