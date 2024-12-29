import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
from transformers import DistilBertTokenizer
import joblib

# # Function to clean text
# def clean_text(text):
#     text = re.sub(r'\W', ' ', text)  # Remove special characters
#     text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
#     text = re.sub(r'\d', '', text)  # Remove numbers
#     text = text.lower().strip()  # Lowercase
#     return text

# Load the data
data_path = './data/Resume/cleaned_resume_data.csv'
data = pd.read_csv(data_path)

# # Clean the text column
data['cleaned_resume'] = data['resume_str']

# Encode labels
label_encoder = LabelEncoder()
data['encoded_category'] = label_encoder.fit_transform(data['category'])

# Save label encoder for deployment
joblib.dump(label_encoder, './models/label_encoder.pkl')

# Save the cleaned data
data[['cleaned_resume', 'encoded_category']].to_csv('./data/cleaned_data.csv', index=False)

print("Data preprocessing complete. Cleaned data saved to './data/cleaned_data.csv'.")
