import streamlit as st
from transformers import pipeline

# --- PAGE CONFIG ---
st.set_page_config(page_title="HRVibeCheck", page_icon="👔")

def main():
    st.title("👔 HRVibeCheck: Smart HR Assistant")
    st.markdown("""
    **Project for ISOM5240** *Target Company:* [Insert Company Name]  
    *Objective:* Maximize retention and automate candidate matching using Deep Learning.
    """)

    st.divider()

    # --- SIDEBAR / INPUT ---
    st.sidebar.header("Candidate Data")
    candidate_name = st.sidebar.text_input("Candidate Name", "John Doe")
    resume_text = st.sidebar.text_area("Paste Resume Text Here", height=300)

    if st.sidebar.button("Run Vibe Check"):
        if resume_text:
            # --- PIPELINE 1: Retention Score (The Fine-tuned Model) ---
            # For now, we use a placeholder pre-trained model
            st.subheader(f"Analysis for {candidate_name}")
            
            with st.spinner('Calculating Retention Vibe...'):
                # We will replace this with your fine-tuned model path later
                retention_pipe = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
                retention_result = retention_pipe(resume_text[:512]) # Truncate for BERT limits
                
                label = retention_result[0]['label']
                score = retention_result[0]['score']
                
                # Logic to map labels to your business case
                status = "STABLE" if label == "POSITIVE" else "LEAVE RISK"
                st.metric("Retention Stability Score", f"{score:.2%}", delta=status)

            # --- PIPELINE 2: Work Match (Skill Extraction) ---
            with st.spinner('Matching Skills...'):
                # Using Zero-Shot as a second pipeline for matching
                match_pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
                candidate_labels = ["Technical", "Managerial", "Creative", "Operational"]
                match_result = match_pipe(resume_text, candidate_labels)
                
                st.write("### Work Match Analysis")
                st.bar_chart({match_result['labels'][i]: match_result['scores'][i] for i in range(len(candidate_labels))})

            st.success("Vibe Check Complete!")
            st.balloons()
        else:
            st.error("Please paste a resume to analyze.")

if __name__ == "__main__":
    main()
