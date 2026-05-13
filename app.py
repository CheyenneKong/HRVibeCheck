import streamlit as st
from transformers import pipeline
import PyPDF2
from docx import Document

# --- PAGE CONFIG ---
st.set_page_config(page_title="HRVibeCheck", page_icon="👔", layout="wide")

# Fixed CSS: Darker text and professional alert-style box
st.markdown("""
    <style>
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e1e4e8; }
    .logic-box { 
        background-color: #f0f7ff; 
        color: #1e3a8a; 
        padding: 20px; 
        border-radius: 10px; 
        border-left: 5px solid #2563eb;
        line-height: 1.6;
    }
    .logic-box b { color: #1e3a8a; }
    .logic-box li { color: #1e40af; }
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
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

@st.cache_resource
def load_pipelines():
    grader_pipe = pipeline("text-classification", model="Cheykong/HRVibeCheck")
    extractor_pipe = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    return grader_pipe, extractor_pipe

grader_pipe, extractor_pipe = load_pipelines()

def main():
    st.title("👔 HRVibeCheck: Smart HR Assistant")
    st.caption("ISOM5240 L2 Group Project | Cheyenne Kong & Janice Ho")

    # --- METHODOLOGY SECTION (Color Fixed) ---
    with st.expander("ℹ️ How the 'Vibe Check' Logic Works", expanded=True):
        st.markdown("""
        <div class="logic-box">
        <b>Deep Learning Architecture:</b> This system utilizes a fine-tuned <b>DistilBERT</b> transformer model. 
        Unlike simple keyword matching, the "Vibe Match Confidence" is calculated using:
        <ul>
            <li><b>Semantic Alignment:</b> The model analyzes the contextual relationship between the Resume and the Job Description (JD).</li>
            <li><b>Sequence Classification:</b> Inputs are processed as a combined pair <code>[Resume] + [SEP] + [JD]</code> to capture fit.</li>
            <li><b>Confidence Score:</b> A Softmax probability output reflecting the model's certainty in the classification.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # --- SIDEBAR INPUTS ---
    st.sidebar.header("📁 Step 1: Candidate Data")
    candidate_name = st.sidebar.text_input("Candidate Name", "John Doe")
    uploaded_resume = st.sidebar.file_uploader("Upload Resume (PDF/Word)", type=["pdf", "docx"])
    resume_manual = st.sidebar.text_area("Or Paste Resume Text", height=150)

    st.sidebar.header("📝 Step 2: Job Requirements")
    jd_text = st.sidebar.text_area("Paste Job Description (JD)", height=200, placeholder="Enter the job requirements here...")

    resume_content = ""
    if uploaded_resume:
        resume_content = extract_text_from_file(uploaded_resume)
    else:
        resume_content = resume_manual

    st.sidebar.divider()
    run_button = st.sidebar.button("🚀 Run Vibe Check Analysis")

    # --- MAIN ANALYSIS ---
    if run_button:
        if not resume_content or not jd_text:
            st.warning("⚠️ Please provide both a Resume and a Job Description to proceed.")
        else:
            with st.status("Performing Deep Learning Analysis...", expanded=True) as status:
                combined_input = f"Resume: {resume_content} [SEP] JD: {jd_text}"
                st.write("Calculating Semantic Fit...")
                retention_result = grader_pipe(combined_input[:512])[0]
                st.write("Extracting Professional Entities...")
                entities = extractor_pipe(resume_content)
                status.update(label="Analysis Complete!", state="complete", expanded=False)

            st.subheader(f"Dashboard: {candidate_name}")
            tab1, tab2, tab3 = st.tabs(["🎯 Fit Analysis", "🔍 Entity Extraction", "📄
