import streamlit as st
from transformers import pipeline
import PyPDF2
from docx import Document

# --- PAGE CONFIG ---
st.set_page_config(page_title="HRVibeCheck", page_icon="👔", layout="wide")

st.markdown("""
    <style>
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e1e4e8; }
    .hr-info-box {
        background-color: #f8fafc;
        color: #334155;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #0f172a;
        line-height: 1.6;
    }
    .hr-info-box b { color: #0f172a; }
    </style>
    """, unsafe_allow_html=True)

# --- UTILITY FUNCTIONS ---
def extract_text_from_file(uploaded_file):
    try:
        if uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            return "".join([page.extract_text() or "" for page in pdf_reader.pages])
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(uploaded_file)
            return "\n".join([para.text for para in doc.paragraphs])
        return None
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

@st.cache_resource
def load_pipelines():
    # Pipeline 1: Updated to your latest fine-tuned model
    grader_pipe = pipeline(
        "text-classification", 
        model="Cheykong/HRVibeCheck-Retention-Predictor"
    )
    
    # Pipeline 2: Skill Extraction (NER)
    extractor_pipe = pipeline(
        "ner", 
        model="dslim/bert-base-NER", 
        aggregation_strategy="simple"
    )
    return grader_pipe, extractor_pipe

grader_pipe, extractor_pipe = load_pipelines()

def main():
    st.title("👔 HRVibeCheck: Smart HR Assistant")
    st.caption("Retention Prediction + Skills Extraction | ISOM5240 Group Project")

    with st.expander("🔍 How HRVibeCheck Works", expanded=True):
        st.markdown("""
        <div class="hr-info-box">
        <b>Two-Pipeline AI System:</b><br>
        • <b>Pipeline 1 (Retention Predictor)</b>: Analyzes the resume to predict if the candidate is likely to stay long-term.<br>
        • <b>Pipeline 2 (Skills Extractor)</b>: Automatically identifies key skills, technologies, and experiences.
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # --- SIDEBAR ---
    st.sidebar.header("📥 Candidate Input")
    candidate_name = st.sidebar.text_input("Candidate Name", "John Doe")

    st.sidebar.subheader("Resume")
    uploaded_resume = st.sidebar.file_uploader("Upload PDF or Word", type=["pdf", "docx"])
    resume_manual = st.sidebar.text_area("Or paste resume text here", height=150)

    st.sidebar.subheader("Job Description (Optional)")
    jd_text = st.sidebar.text_area("Paste Job Description", height=120, 
                                   placeholder="Optional: Helps improve context...")

    resume_content = ""
    if uploaded_resume:
        resume_content = extract_text_from_file(uploaded_resume)
    else:
        resume_content = resume_manual

    run_button = st.sidebar.button("🚀 Analyze Candidate", type="primary")

    # --- MAIN ANALYSIS ---
    if run_button:
        if not resume_content:
            st.warning("⚠️ Please upload a resume or paste resume text.")
        else:
            with st.status("Analyzing candidate...", expanded=True) as status:
                # Pipeline 1: Retention Prediction
                status.write("Running Retention Prediction (Pipeline 1)...")
                input_text = resume_content[:512]  # DistilBERT limit
                retention_result = grader_pipe(input_text)[0]

                # Pipeline 2: Skills Extraction
                status.write("Extracting key skills (Pipeline 2)...")
                entities = extractor_pipe(resume_content[:1000])

                status.update(label="Analysis Complete!", state="complete", expanded=False)

            # --- RESULTS ---
            st.subheader(f"📊 Analysis for: {candidate_name}")

            col1, col2 = st.columns(2)
            with col1:
                label = retention_result['label']
                score = retention_result['score']
                is_hire = label in ["LABEL_1", "HIRE", "positive"]

                st.metric(
                    label="Retention / Hire Probability",
                    value=f"{score:.1%}",
                    delta="HIGH RETENTION POTENTIAL" if is_hire else "REVIEW RECOMMENDED",
                    delta_color="normal" if is_hire else "inverse"
                )

            with col2:
                st.write("**Extracted Key Skills**")
                skills = [e['word'] for e in entities if e['entity_group'] in ['ORG', 'MISC']]
                unique_skills = list(set(skills))
                if unique_skills:
                    for skill in unique_skills[:8]:
                        st.success(f"• {skill}")
                else:
                    st.write("No major skills detected.")

            st.divider()
            st.write("**Resume Preview:**")
            st.info(resume_content[:800] + "..." if len(resume_content) > 800 else resume_content)

if __name__ == "__main__":
    main()
