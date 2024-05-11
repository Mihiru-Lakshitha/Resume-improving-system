import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import PyPDF2
import re
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load the trained model
model_path = 'model.h5'
model = load_model(model_path)

# Load the tokenizer
tokenizer_path = 'tokenizer.pkl'  # Assuming you saved the tokenizer as a pickle file
with open(tokenizer_path, 'rb') as file:
    tokenizer = pickle.load(file)

# Function to preprocess text
def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)
    
    # Lowercasing
    tokens = [word.lower() for word in tokens]
    
    # Remove any further punctuation and special characters
    tokens = [word for word in tokens if word.isalnum()]
    
    # Remove and further stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Join the tokens back into a single string
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file_path):
    text = ""
    with open(pdf_file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)
        
        for page_num in range(num_pages):
            page = pdf_reader.pages[0]
            text += page.extract_text()
            
    return text

# Function to clean resume
def clean_resume(resume_text):
    resume_text = re.sub('http\S+\s*', ' ', resume_text)  # remove URLs
    resume_text = re.sub('RT|cc', ' ', resume_text)  # remove RT and cc
    resume_text = re.sub('#\S+', '', resume_text)  # remove hashtags
    resume_text = re.sub('@\S+', '  ', resume_text)  # remove mentions
    resume_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resume_text)  # remove punctuations
    resume_text = re.sub(r'[^\x00-\x7f]', r' ', resume_text) 
    resume_text = re.sub('\s+', ' ', resume_text)  # remove extra whitespace
    return resume_text

# Define a function to make predictions
def predict_class(text):
    cleaned_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)
    predicted_probabilities = model.predict(padded_sequence)
    predicted_class_index = np.argmax(predicted_probabilities)
    predicted_class_label = label_encoder.classes_[predicted_class_index]
    return predicted_class_label

# Define the Streamlit app
def main():
    st.title("Resume Analyzer")
    st.write("Upload your resume and analyze it!")

    # Upload the resume file
    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:
        file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}
        st.write(file_details)
        
        # Read the file
        file_type = uploaded_file.type.split('/')[1]
        if file_type == 'pdf':
            resume_text = extract_text_from_pdf(uploaded_file)
        elif file_type == 'txt':
            resume_text = uploaded_file.read().decode('utf-8')
        else:
            st.error("Unsupported file format. Please upload a PDF or TXT file.")
            return

        # Clean and preprocess the resume text
        cleaned_resume_text = clean_resume(resume_text)

        # Display the cleaned resume text
        st.subheader("Cleaned Resume Text:")
        st.write(cleaned_resume_text)

        # Make prediction
        predicted_class = predict_class(cleaned_resume_text)
        st.subheader("Predicted Class:")
        st.write(predicted_class)

# Run the Streamlit app
if __name__ == "__main__":
    main()
