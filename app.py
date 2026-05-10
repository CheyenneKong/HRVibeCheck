import streamlit as st
from transformers import pipeline

# --- PAGE CONFIG ---
st.set_page_config(page_title="HRVibeCheck", page_icon="👔")

# --- LOAD MODELS (Outside main to cache them) ---
@st.cache_resource
def load_pipelines():
    # REPLACE "your-username/hr-vibe-grader" with your actual HF path
    grader_pipe = pipeline("text-classification", model="Cheykong/HRVibeCheck")
    
    # This stays the same
    extractor_pipe = pipeline("ner", model="distilbert-base-uncased", aggregation_strategy="simple")
    
    return grader_pipe, extractor_pipe

grader_pipe, extractor_pipe = load_pipelines()

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
            st.subheader(f"Analysis for {candidate_name}")
            
            # --- PIPELINE 1: The Grader (Fine-tuned Model) ---
            with st.spinner('Calculating Retention Vibe...'):
                # Truncate text to 512 tokens for BERT safety
                retention_result = grader_pipe(resume_text[:512]) 
                
                # In your model, LABEL_0 = Bad Fit, LABEL_1 = Good Fit (usually)
                raw_label = retention_result[0]['label']
                score = retention_result[0]['score']
                
                # Mapping your model's labels to business vibe
                # If your model outputs "LABEL_1" for success:
                if raw_label == "LABEL_1":
                    status = "HIGH POTENTIAL"
                    delta_color = "normal"
                else:
                    status = "MATCH RISK"
                    delta_color = "inverse"
                
                st.metric("Fit Confidence Score", f"{score:.2%}", delta=status, delta_color=delta_color)

            # --- PIPELINE 2: Extraction (Entity Recognition) ---
            with st.spinner('Extracting Key Entities...'):
                entities = extractor_pipe(resume_text)
                
                st.write("### Key Entities Found")
                # Grouping entities for a cleaner look
                orgs = list(set([ent['word'] for ent in entities if ent['entity_group'] == 'ORG']))
                locs = list(set([ent['word'] for ent in entities if ent['entity_group'] == 'LOC']))
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Organizations/Universities:**")
                    for org in orgs[:5]: st.write(f"- {org}")
                with col2:
                    st.write("**Locations:**")
                    for loc in locs[:5]: st.write(f"- {loc}")

            st.success("Vibe Check Complete!")
            st.balloons()
        else:
            st.error("Please paste a resume to analyze.")

if __name__ == "__main__":
    main()
