import streamlit as st
from transformers import pipeline
import PyPDF2
from docx import Document
import io

# --- PAGE CONFIG ---
st.set_page_config(page_title="HRVibeCheck", page_icon="👔", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e1e4e8; }
    </style>
    """, unsafe_allow_html=True)

# --- UTILITY FUNCTIONS ---
def extract_text_from_file(uploaded_file):
    """Extracts text from PDF or DOCX files."""
    try:
        if uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
            return text
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(uploaded_file)
            return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

@st.cache_resource
def load_pipelines():
    """Loads the fine-tuned grader and the NER extractor."""
    # Pipeline 1: Your Fine-tuned Model from Hugging Face
    model_path = "Cheykong/HRVibeCheck" 
    grader_pipe = pipeline("text-classification", model=model_path)
    
    # Pipeline 2: NER (Standard Professional Entity Extraction)
    extractor_pipe = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    
    return grader_pipe, extractor_pipe

# Load models once
grader_pipe, extractor_pipe = load_pipelines()

# --- MAIN APP ---
def main():
    st.title("👔 HRVibeCheck: Smart HR Assistant")
    st.caption("ISOM5240 L2 | Cheyenne Kong & Janice Ho")

    # --- SIDEBAR: INPUT ---
    st.sidebar.header("📥 Input Sources")
    candidate_name = st.sidebar.text_input("Candidate Name", "John Doe")
    
    uploaded_file = st.sidebar.file_uploader("Upload Resume (PDF/Word)", type=["pdf", "docx"])
    manual_text = st.sidebar.text_area("Or Paste Resume Text Manually", height=200)

    # Text Logic
    resume_text = ""
    if uploaded_file is not None:
        resume_text = extract_text_from_file(uploaded_file)
    elif manual_text:
        resume_text = manual_text

    st.sidebar.divider()
    run_button = st.sidebar.button("🚀 Run Vibe Check")

    # --- MAIN DISPLAY ---
    if run_button:
        if not resume_text:
            st.error("Please provide a resume by uploading a file or pasting text.")
        else:
            with st.status("Analyzing Candidate...", expanded=True) as status:
                st.write("Running Vibe Grader...")
                retention_result = grader_pipe(resume_text[:512])[0] 
                
                st.write("Extracting Professional Entities...")
                entities = extractor_pipe(resume_text)
                status.update(label="Analysis Complete!", state="complete", expanded=False)

            st.subheader(f"Analysis Results: {candidate_name}")
            
            # --- TABS (Fixed Syntax Here) ---
            tab1, tab2, tab3 = st.tabs(["🎯 Match Scoring", "🔍 Entity Extraction", "📄 Source Text"])

            with tab1:
                col1, col2 = st.columns(2)
                label = retention_result['label']
                score = retention_result['score']
                is_match = (label == "LABEL_1")
                
                with col1:
                    st.metric(label="Confidence", value=f"{score:.2%}", 
                              delta="HIGH POTENTIAL" if is_match else "MATCH RISK", 
                              delta_color="normal" if is_match else "inverse")
                with col2:
                    st.write("**Visual Match Score**")
                    st.progress(score)

            with tab2:
                orgs = sorted(list(set([e['word'] for e in entities if e['entity_group'] == 'ORG'])))
                locs = sorted(list(set([e['word'] for e in entities if e['entity_group'] == 'LOC'])))
                
                c1, c2 = st.columns(2)
                with c1:
                    st.write("**Organizations Detected:**")
                    for org in orgs[:8]: st.info(f"🏛️ {org}")
                with c2:
                    st.write("**Locations Detected:**")
                    for loc in locs[:8]: st.success(f"📍 {loc}")

            with tab3:
                st.text_area("Extracted Content", value=resume_text, height=300)

            st.balloons()

if __name__ == "__main__":
    main()
