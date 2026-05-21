# 👔 HRVibeCheck - AI-Powered Resume Screening & Hiring Assistant

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![HuggingFace](https://img.shields.io/badge/🤗_Hugging_Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

---

## 🎯 Project Objective

**HRVibeCheck** is a smart AI hiring assistant that automates resume screening by intelligently matching candidates with job descriptions.

It solves a major pain point in recruitment: **manual resume review is time-consuming and subjective**. This application uses deep learning to provide fast, consistent, and data-driven hiring recommendations.

**Target Company**: HRVibeCheck (AI Talent Acquisition Platform)  
**Website**: [https://hrvibecheck.com](https://hrvibecheck.com) *(fictional for project)*

---

## ✨ Key Features

- Upload **multiple PDF/DOCX resumes** at once
- Fine-tuned model gives **Hire Recommendation Score** (0–100%)
- Automatic **Key Skills Extraction** using NER (no predefined list)
- Beautiful ranked dashboard with skill visualization
- Resume text preview
- Fast processing with progress tracking

---

## 🧠 Model Pipelines (Required for ISOM5240)

| Pipeline | Task                        | Model ID                                              | Type                    | Status          |
|----------|-----------------------------|-------------------------------------------------------|-------------------------|-----------------|
| **P1**   | Hire Recommendation         | `Cheykong/HRVibeCheck-Hire-Recommendation-Model`     | Fine-tuned Text Classification | ✅ Trained & Pushed |
| **P2**   | Skill Extraction (NER)      | `algiraldohe/lm-ner-linkedin-skills-recognition`     | Token Classification    | ✅ Best performer |

---

## 🚀 Live Demo

**Streamlit Cloud App**: [https://hrvibecheck.streamlit.app/](https://hrvibecheck.streamlit.app/)  

**Fine-tuned Model on Hugging Face**:  
[https://huggingface.co/Cheykong/HRVibeCheck-Hire-Recommendation-Model](https://huggingface.co/Cheykong/HRVibeCheck-Hire-Recommendation-Model)

---

## 🛠️ How to Run Locally

### Prerequisites
- Python 3.9+

### Installation

```bash
git clone https://github.com/Cheykong/HRVibeCheck.git
cd HRVibeCheck

pip install -r requirements.txt
