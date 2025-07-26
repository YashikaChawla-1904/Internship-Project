import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    # Combine job description with resumes
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()

    # Calculate cosine similarity
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()
    
    return cosine_similarities

# Streamlit app
st.title("AI Resume Screening & Candidate Ranking System")

# Login functionality
def login():
    st.sidebar.header("Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    
    if st.sidebar.button("Login"):
        if username == "admin" and password == "password":  # Replace with your own logic
            st.session_state['logged_in'] = True
            st.success("Logged in successfully!")
        else:
            st.error("Invalid username or password")

# Check if user is logged in
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    login()
else:
    # Job description input
    st.header("Job Description")
    job_description = st.text_area("Enter the job description")

    # File uploader
    st.header("Upload Resumes")
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

    if uploaded_files and job_description:
        st.header("Ranking Resumes")

        resumes = []
        for file in uploaded_files:
            text = extract_text_from_pdf(file)
            resumes.append(text)

        # Rank resumes
        scores = rank_resumes(job_description, resumes)

        # Debugging step: print the lengths
        print(f"Length of uploaded_files: {len(uploaded_files)}")
        print(f"Length of scores: {len(scores)}")

        # Ensure both lists have the same length
        if len(uploaded_files) != len(scores):
            if len(uploaded_files) > len(scores):
                scores.extend([None] * (len(uploaded_files) - len(scores)))
            elif len(uploaded_files) < len(scores):
                scores = scores[:len(uploaded_files)]

        # Now create the DataFrame
        results = pd.DataFrame({"Resumes": [file.name for file in uploaded_files], "Score": scores})

        # Optional: print the results to confirm
        print(results)
        results = results.sort_values(by="Score", ascending=False)
        st.write(results)
    else:
        st.warning("Please upload resumes and enter a job description to see the ranking.")