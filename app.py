import streamlit as st
from transformers import pipeline
import PyPDF2
from docx import Document

st.set_page_config(page_title="HRVibeCheck", page_icon="👔", layout="wide")

# --- LIGHTWEIGHT PIPELINES ---
@st.cache_resource(show_spinner="Loading AI models...")
def load_pipelines():
    # Pipeline 1: Your fine-tuned model
    retention_pipe = pipeline(
        "text-classification", 
        model="Cheykong/HRVibeCheck-Retention-Predictor",
        device=-1   # Force CPU to avoid timeout
    )
    
    # Pipeline 2: Very light zero-shot model
    skill_pipe = pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/deberta-v3-base-zeroshot-v2.0",   # Much lighter
        device=-1
    )
    return retention_pipe, skill_pipe

retention_pipe, skill_pipe = load_pipelines()

def extract_text_from_file(uploaded_file):
    try:
        if uploaded_file.type == "application/pdf":
            pdf = PyPDF2.PdfReader(uploaded_file)
            return "".join([page.extract_text() or "" for page in pdf.pages])
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(uploaded_file)
            return "\n".join([p.text for p in doc.paragraphs])
        return None
    except:
        return None

# --- MAIN ---
def main():
    st.title("👔 HRVibeCheck")
    st.caption("Retention Prediction + Skills Extraction")

    uploaded_file = st.file_uploader("Upload Resume (PDF or Word)", type=["pdf", "docx"])
    resume_text = st.text_area("Or paste resume text here", height=250)

    if st.button("🚀 Analyze Candidate", type="primary") and (uploaded_file or resume_text):
        text = extract_text_from_file(uploaded_file) if uploaded_file else resume_text

        with st.spinner("Analyzing with AI..."):
            # Pipeline 1
            result1 = retention_pipe(text[:512])[0]
            # Pipeline 2
            labels = ["Python", "SQL", "Machine Learning", "AWS", "Docker", "Leadership", 
                     "Project Management", "Data Analysis", "PyTorch", "Communication"]
            result2 = skill_pipe(text[:1000], labels, multi_label=True)

        st.success("Analysis Complete!")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Retention Probability", f"{result1['score']:.1%}")
        with col2:
            st.write("**Key Skills Found**")
            skills = [lab for lab, sc in zip(result2['labels'], result2['scores']) if sc > 0.4]
            for s in skills[:8]:
                st.success(f"• {s}")

        st.info("**Resume Preview:**\n" + text[:700] + "..." if len(text) > 700 else text)

if __name__ == "__main__":
    main()
