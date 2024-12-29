import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import joblib
# import re
import os
# from clean_resume import clean_text

# # Function to clean text
# def clean_text(text):
#     text = re.sub(r'\\n|\\t', ' ', text)  # Remove newline/tab characters
#     text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
#     text = re.sub(r'[\W]+', ' ', text)  # Remove special characters
#     return text.strip()

# Load the fine-tuned model and tokenizer
MODEL_PATH = "./models/resume_classifier_model"
LABEL_ENCODER_PATH = "./models/label_encoder.pkl"

# print("MODEL_PATH:", os.path.abspath(MODEL_PATH))
# print("LABEL_ENCODER_PATH:", os.path.abspath(LABEL_ENCODER_PATH))

label_encoder = joblib.load(LABEL_ENCODER_PATH)
# @st.cache_resource()
def load_model():
    print("Loading model, tokenizer, and label encoder...")
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    print("Model, tokenizer, and label encoder loaded successfully.")
    return model, tokenizer, label_encoder

model, tokenizer,label_encoder = load_model()

print("Model loaded succesfully")

# Function for prediction
def predict_resume_category(text):
    # Tokenize input text
    cleaned_text = clean_text(text)
    inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=-1).item()
    return label_encoder.inverse_transform([predicted_class_id])[0]


# Streamlit UI
st.title("Resume Category Classifier")
st.write("Enter a resume and the model will predict its category.")

# Input box for user text
user_input = st.text_area("Paste Resume Text Here", height=300)

# Predict button
if st.button("Predict Category"):
    if user_input.strip() != "":
        st.write("Predicting...")
        predicted_category = predict_resume_category(user_input)
        st.success(f"Predicted Category: **{predicted_category}**")
    else:
        st.error("Please enter some text for prediction.")
