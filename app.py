import streamlit as st
from transformers import pipeline

# --- PAGE CONFIG ---
st.set_page_config(page_title="HRVibeCheck", page_icon="👔", layout="wide")

# Custom CSS for a professional look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e1e4e8; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_pipelines():
    grader_pipe = pipeline("text-classification", model="Cheykong/HRVibeCheck")
    # Using a specialized NER model for better results
    extractor_pipe = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    return grader_pipe, extractor_pipe

grader_pipe, extractor_pipe = load_pipelines()

def main():
    # --- HEADER ---
    col_title, col_logo = st.columns([4, 1])
    with col_title:
        st.title("👔 HRVibeCheck: Smart HR Assistant")
        st.caption("Empowering ISOM5240 L2 - Candidate Analysis Suite")

    # --- SIDEBAR ---
    st.sidebar.header("📥 Input Candidate Data")
    candidate_name = st.sidebar.text_input("Candidate Name", "John Doe")
    resume_text = st.sidebar.text_area("Paste Resume Text Here", height=400)
    
    # User-set threshold
    threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.7)

    if st.sidebar.button("🚀 Run Deep Analysis"):
        if resume_text:
            # --- PROCESSING ---
            with st.status("Analyzing Vibe...", expanded=True) as status:
                st.write("Checking model versioning...")
                retention_result = grader_pipe(resume_text[:512])[0]
                
                st.write("Extracting professional entities...")
                entities = extractor_pipe(resume_text)
                status.update(label="Analysis Complete!", state="complete", expanded=False)

            # --- MAIN DASHBOARD ---
            st.subheader(f"Results for {candidate_name}")
            
            # Use Tabs to organize information
            tab1, tab2, tab3 = st.tabs(["🎯 Fit Analysis", "🏢 Professional Profile", "📄 Original Text"])

            with tab1:
                col_m1, col_m2 = st.columns(2)
                
                raw_label = retention_result['label']
                score = retention_result['score']
                
                is_match = raw_label == "LABEL_1"
                status_text = "HIGH POTENTIAL" if is_match else "MATCH RISK"
                
                with col_m1:
                    st.metric("Retention Stability Score", f"{score:.2%}", delta=status_text, 
                              delta_color="normal" if is_match else "inverse")
                
                with col_m2:
                    # Visual Indicator Bar
                    color = "green" if score > threshold else "orange"
                    st.write(f"**Confidence Level:**")
                    st.progress(score)
                    if score < threshold:
                        st.warning("⚠️ This score is below the selected threshold.")

            with tab2:
                st.write("### Key Information Extracted")
                # Grouping entities
                orgs = sorted(list(set([ent['word'] for ent in entities if ent['entity_group'] == 'ORG'])))
                locs = sorted(list(set([ent['word'] for ent in entities if ent['entity_group'] == 'LOC'])))
                
                c1, c2 = st.columns(2)
                with c1:
                    st.write("**Organizations & Education**")
                    if orgs:
                        for org in orgs: st.info(f"🏛️ {org}")
                    else: st.write("None detected.")
                
                with c2:
                    st.write("**Geographic Context**")
                    if locs:
                        for loc in locs: st.success(f"📍 {loc}")
                    else: st.write("None detected.")

            with tab3:
                with st.expander("Show/Hide Resume Content"):
                    st.text(resume_text)

            st.balloons()
        else:
            st.error("Please paste a resume to begin analysis.")

if __name__ == "__main__":
    main()
