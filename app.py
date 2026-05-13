import streamlit as st
from transformers import pipeline
import PyPDF2
from docx import Document
import io

# 1. Add this function at the top of app.py
def extract_text_from_file(uploaded_file):
    try:
        if uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(uploaded_file)
            return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

# ... (Keep your load_pipelines code here) ...

def main():
    st.title("👔 HRVibeCheck")
    
    # 2. Update the Sidebar logic
    st.sidebar.header("Step 1: Upload Resume")
    uploaded_file = st.sidebar.file_uploader("Choose a PDF or Word file", type=["pdf", "docx"])
    
    # Initialize resume_text as empty
    resume_text = ""

    if uploaded_file is not None:
        resume_text = extract_text_from_file(uploaded_file)
        if resume_text:
            st.sidebar.success("File content extracted!")
            # Optional: Show a preview of the extracted text
            with st.sidebar.expander("View Extracted Text"):
                st.write(resume_text[:500] + "...")
    else:
        # Fallback to manual text area if no file is uploaded
        resume_text = st.sidebar.text_area("Or paste text manually", height=200)

    # 3. The Run Button
    if st.sidebar.button("Run Vibe Check"):
        if resume_text:
            # ... (Your existing pipeline 1 and pipeline 2 code) ...
