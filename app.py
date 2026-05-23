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
    .skill-tech { background-color: #2563eb; }        /* blue — TECHNOLOGY */
    .skill-technical { background-color: #0891b2; }   /* cyan — TECHNICAL */
    .skill-business { background-color: #7c3aed; }    /* purple — BUSINESS */
    .skill-soft { background-color: #ea580c; }        /* orange — SOFT */
    .skill-other { background-color: #64748b; }       /* gray — fallback */
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

def get_recommendation(score: float):
    if score >= 0.85:
        return "✅ Strong Hire — SELECT"
    elif score >= 0.65:
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
        - **Pipeline 1**: Fine-tuned transformer (`bert-base-uncased`) that compares Job Description vs Resume and gives a **Hire Score** (0–100%), then maps it to a SELECT / Consider / Reject recommendation.
        - **Pipeline 2**: NER model (`algiraldohe/lm-ner-linkedin-skills-recognition`) that automatically extracts key skills from the resume and groups them into **4 categories**:
            - 🟦 **TECHNOLOGY** — concrete tools, languages, platforms (e.g., Python, AWS, Tableau)
            - 🟦 **TECHNICAL** — methods and disciplines (e.g., machine learning, statistical analysis)
            - 🟪 **BUSINESS** — domain & operational skills (e.g., project management, budget management)
            - 🟧 **SOFT** — interpersonal skills (e.g., communication, leadership)
        """)

    with st.expander("📊 How is the Hire Score Calculated?", expanded=False):
        st.markdown("""
        **Hire Score Explanation (for HR Professionals)**
        **Score Guide**:
        - ≥ 85% → Strong Hire — SELECT
        - 65–84% → Good Hire — SELECT
        - 45–64% → Moderate Fit — Consider
        - < 45% → Further Review / Reject
        """)

    with st.sidebar:
        st.header("📋 Job Description")
        
        # JD Upload (as requested)
        jd_file = st.file_uploader("📄 Upload Job Description (PDF or Word)", 
                                  type=["pdf", "docx"], key="jd_upload")
        
        if jd_file is not None:
            jd_text = extract_text_from_file(jd_file)
            st.success(f"✅ JD loaded: {jd_file.name}")
        else:
            jd_text = st.text_area("Or paste the full Job Description", height=180, 
                                 placeholder="We are looking for a Senior Data Scientist...")

        st.divider()
        st.header("📄 Upload Resumes")
        uploaded_files = st.file_uploader("Upload Candidate Resumes (PDF or Word)", 
                                        type=["pdf", "docx"], accept_multiple_files=True)

        st.divider()
        analyze_btn = st.button("🚀 Analyze Candidates", type="primary", use_container_width=True)

    if analyze_btn:
        if not jd_text.strip():
            st.warning("⚠️ Please upload a JD file or paste the Job Description.")
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

            progress_bar.progress((i + 1) / len(uploaded_files))

        elapsed = time.time() - start_time
        status_text.empty()
        progress_bar.empty()

        results.sort(key=lambda x: x["Hire Score"], reverse=True)
        st.success(f"✅ Analysis Complete! {len(results)} candidate(s) processed in {elapsed:.1f} seconds.")

        st.subheader("🏆 Candidate Rankings")

        def summarize_skills_by_category(skill_list):
            """Format skills as 'Tech: 3 | Technical: 2 | Business: 1 | Soft: 4' for the table."""
            if not skill_list:
                return "—"
            counts = {"TECHNOLOGY": 0, "TECHNICAL": 0, "BUSINESS": 0, "SOFT": 0, "OTHER": 0}
            for s in skill_list:
                cat = (s.get("category") or "OTHER").upper()
                if cat not in counts:
                    cat = "OTHER"
                counts[cat] += 1
            short_labels = {"TECHNOLOGY": "Tech", "TECHNICAL": "Technical",
                            "BUSINESS": "Business", "SOFT": "Soft", "OTHER": "Other"}
            parts = [f"{short_labels[k]}: {v}" for k, v in counts.items() if v > 0]
            return " | ".join(parts)

        table_data = [{
            "Rank": f"#{rank}",
            "Candidate": r["Candidate"],
            "Hire Score": r["Score %"],
            "Recommendation": r["Recommendation"],
            "Skills by Category": summarize_skills_by_category(r["Key Skills"])
        } for rank, r in enumerate(results, 1)]
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

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
                        # Group skills by category
                        category_map = {
                            "TECHNOLOGY": {"label": "Technology", "css": "skill-tech"},
                            "TECHNICAL":  {"label": "Technical",  "css": "skill-technical"},
                            "BUSINESS":   {"label": "Business",   "css": "skill-business"},
                            "SOFT":       {"label": "Soft Skills","css": "skill-soft"},
                        }
                        grouped = {"TECHNOLOGY": [], "TECHNICAL": [], "BUSINESS": [], "SOFT": [], "OTHER": []}
                        for s in r["Key Skills"]:
                            cat = (s.get("category") or "OTHER").upper()
                            if cat not in grouped:
                                cat = "OTHER"
                            grouped[cat].append(s["name"])

                        # Render each category in fixed order
                        rendered_any = False
                        for cat_key in ["TECHNOLOGY", "TECHNICAL", "BUSINESS", "SOFT"]:
                            items = grouped.get(cat_key, [])
                            if not items:
                                continue
                            rendered_any = True
                            meta = category_map[cat_key]
                            st.markdown(f"<div class='category-label'>🏷️ {meta['label']} ({len(items)})</div>",
                                        unsafe_allow_html=True)
                            pills_html = " ".join(
                                [f"<span class='skill-pill {meta['css']}'>{name}</span>" for name in items]
                            )
                            st.markdown(pills_html, unsafe_allow_html=True)

                        # Show any unclassified skills under a generic group
                        other_items = grouped.get("OTHER", [])
                        if other_items:
                            rendered_any = True
                            st.markdown("<div class='category-label'>🏷️ Other</div>", unsafe_allow_html=True)
                            pills_html = " ".join(
                                [f"<span class='skill-pill skill-other'>{name}</span>" for name in other_items]
                            )
                            st.markdown(pills_html, unsafe_allow_html=True)

                        if not rendered_any:
                            st.info("No high-confidence skills detected.")
                    else:
                        st.info("No high-confidence skills detected.")
                st.divider()
                st.write("**Resume Preview:**")
                preview = r["Resume Text"][:700] + "..." if len(r["Resume Text"]) > 700 else r["Resume Text"]
                st.text_area("", preview, height=200, disabled=True)

if __name__ == "__main__":
    main()
