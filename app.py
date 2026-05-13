import streamlit as st
from transformers import pipeline
import PyPDF2
from docx import Document

# --- PAGE CONFIG ---
st.set_page_config(page_title="HRVibeCheck", page_icon="👔", layout="wide")

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
    # Pipeline 1: Your Fine-tuned Model
    grader_pipe = pipeline("text-classification", model="Cheykong/HRVibeCheck")
    # Pipeline 2: NER
    extractor_pipe = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    return grader_pipe, extractor_pipe

grader_pipe, extractor_pipe = load_pipelines()

def main():
    st.title("👔 HRVibeCheck: Match Analysis")
    
    # --- SIDEBAR: TWO INPUTS ---
    st.sidebar.header("1. The Candidate")
    uploaded_resume = st.sidebar.file_uploader("Upload Resume", type=["pdf", "docx"])
    resume_manual = st.sidebar.text_area("Or Paste Resume", height=150)
    
    st.sidebar.header("2. The Requirement")
    jd_text = st.sidebar.text_area("Paste Job Description (JD) here", height=150, placeholder="What are you looking for?")

    # Extract Resume Text
    resume_content = ""
    if uploaded_resume:
        resume_content = extract_text_from_file(uploaded_resume)
    else:
        resume_content = resume_manual

    st.sidebar.divider()
    run_button = st.sidebar.button("🚀 Run Vibe Check")

    # --- MAIN DISPLAY ---
    if run_button:
        if not resume_content or not jd_text:
            st.error("Please provide both a Resume and a Job Description to compare.")
        else:
            with st.status("Comparing Resume to JD...", expanded=True) as status:
                # COMBINE TEXT: This mimics your Colab training flow
                # We put a separator so the model knows where the Resume ends and JD begins
                combined_input = f"Resume: {resume_content} [SEP] JD: {jd_text}"
                
                # Pipeline 1: The Grader
                # Truncate to 512 for BERT safety
                retention_result = grader_pipe(combined_input[:512])[0]
                
                # Pipeline 2: NER (Usually run on Resume only)
                entities = extractor_pipe(resume_content)
                status.update(label="Match Analysis Complete!", state="complete")

            # --- RESULTS ---
            st.subheader("Match Analysis Results")
            tab1, tab2 = st.tabs(["🎯 Match Score", "🔍 Entity Highlights"])

            with tab1:
                label = retention_result['label']
                score = retention_result['score']
                is_match = (label == "LABEL_1")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Vibe Match Confidence", f"{score:.2%}", 
                              delta="GOOD FIT" if is_match else "POOR FIT",
                              delta_color="normal" if is_match else "inverse")
                with col2:
                    st.write("**Fit Visualizer**")
                    st.progress(score)

            with tab2:
                # Group and display entities from resume
                orgs = sorted(list(set([e['word'] for e in entities if e['entity_group'] == 'ORG'])))
                st.write("**Top Organizations in Resume:**")
                st.info(", ".join(orgs[:10]) if orgs else "None detected")

if __name__ == "__main__":
    main()
