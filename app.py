import streamlit as st
from transformers import pipeline
import PyPDF2
from docx import Document

# --- PAGE CONFIG ---
st.set_page_config(page_title="HRVibeCheck", page_icon="👔", layout="wide")

# HR-Friendly Theme: Clean, Professional Blues
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
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

@st.cache_resource
def load_pipelines():
    # Model 1: The "Vibe" Matcher
    grader_pipe = pipeline("text-classification", model="Cheykong/HRVibeCheck")
    # Model 2: Key Info Extractor
    extractor_pipe = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    return grader_pipe, extractor_pipe

grader_pipe, extractor_pipe = load_pipelines()

def main():
    st.title("👔 HRVibeCheck: Smart HR Assistant")
    st.caption("Advanced Candidate Matching Tool | ISOM5240 L2")

    # --- HR-FRIENDLY METHODOLOGY ---
    with st.expander("🔍 Understanding the Vibe Check Analysis", expanded=True):
        st.markdown("""
        <div class="hr-info-box">
        <b>How we assess 'Fit':</b> Unlike traditional keyword tools that just look for specific words, 
        our AI uses <b>Contextual Intelligence</b> to understand a candidate's background.
        <ul>
            <li><b>Beyond Keywords:</b> The system reads the <i>meaning</i> behind professional experiences, identifying how skills in a resume align with the specific needs of your Job Description.</li>
            <li><b>Relationship Scoring:</b> We analyze the candidate and the job requirements simultaneously to see if the "professional vibe" matches your company's retention goals.</li>
            <li><b>Match Confidence:</b> This represents the statistical likelihood that this candidate is a 'High Potential' fit based on patterns learned from successful historical hires.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # --- SIDEBAR ---
    st.sidebar.header("📥 Data Input")
    candidate_name = st.sidebar.text_input("Candidate Name", "John Doe")
    
    st.sidebar.subheader("Step 1: The Resume")
    uploaded_resume = st.sidebar.file_uploader("Upload PDF or Word", type=["pdf", "docx"])
    resume_manual = st.sidebar.text_area("Or Paste Resume Text", height=100)

    st.sidebar.subheader("Step 2: The Job")
    jd_text = st.sidebar.text_area("Paste Job Description", height=150, placeholder="What are the requirements for this role?")

    resume_content = ""
    if uploaded_resume:
        resume_content = extract_text_from_file(uploaded_resume)
    else:
        resume_content = resume_manual

    st.sidebar.divider()
    run_button = st.sidebar.button("🚀 Analyze Candidate Fit")

    # --- ANALYSIS DISPLAY ---
    if run_button:
        if not resume_content or not jd_text:
            st.warning("⚠️ Please provide both a Resume and a Job Description to generate a score.")
        else:
            with st.status("AI is reviewing candidate background...", expanded=True) as status:
                combined_input = f"Resume: {resume_content} [SEP] JD: {jd_text}"
                st.write("Reading professional context...")
                retention_result = grader_pipe(combined_input[:512])[0]
                st.write("Identifying key organizations and locations...")
                entities = extractor_pipe(resume_content)
                status.update(label="Review Complete!", state="complete", expanded=False)

            st.subheader(f"Recruitment Insights: {candidate_name}")
            tab1, tab2, tab3 = st.tabs(["🎯 Match Assessment", "🏢 Candidate Profile", "📄 Original Documents"])

            with tab1:
                label = retention_result['label']
                score = retention_result['score']
                is_high_vibe = (label == "LABEL_1")

                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.metric(
                        label="Overall Match Confidence", 
                        value=f"{score:.2%}", 
                        delta="HIGH POTENTIAL" if is_high_vibe else "FURTHER REVIEW NEEDED",
                        delta_color="normal" if is_high_vibe else "inverse"
                    )
                with col_m2:
                    st.write("**Candidate Alignment Level**")
                    st.progress(score)
                    st.caption("Higher percentages indicate stronger alignment between candidate experience and job needs.")

            with tab2:
                # Group entities for HR profile
                orgs = sorted(list(set([e['word'] for e in entities if e['entity_group'] == 'ORG'])))
                locs = sorted(list(set([e['word'] for e in entities if e['entity_group'] == 'LOC'])))
                
                c1, c2 = st.columns(2)
                with c1:
                    st.write("**Relevant Organizations/Schools:**")
                    if orgs:
                        for org in orgs[:6]: st.info(f"🏢 {org}")
                    else: st.write("No major organizations identified.")
                with c2:
                    st.write("**Geographic Focus:**")
                    if locs:
                        for loc in locs[:6]: st.success(f"📍 {loc}")
                    else: st.write("No locations identified.")

            with tab3:
                st.write("**Resume Content:**")
                st.info(resume_content[:1000] + "..." if len(resume_content) > 1000 else resume_content)
                st.write("**Job Description:**")
                st.info(jd_text)

            st.balloons()

if __name__ == "__main__":
    main()
