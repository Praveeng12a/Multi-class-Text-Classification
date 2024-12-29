import pandas as pd
import re
from bs4 import BeautifulSoup

# Load the dataset
file_path = './data/Resume/Resume.csv'
df = pd.read_csv(file_path)

# Function to clean text
def clean_text(text):
    text = re.sub(r'\\n|\\t', ' ', text)  # Remove newline/tab characters
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'[\W]+', ' ', text)  # Remove special characters
    return text.strip()

# Function to clean HTML content
def clean_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text(separator=" ")  # Extract text from HTML
    return clean_text(text)

# Clean Resume_str column
df['cleaned_resume'] = df['Resume_str'].apply(clean_text)

# Clean Resume_html column
df['cleaned_resume_html'] = df['Resume_html'].apply(clean_html)

# Standardize column names
df.columns = df.columns.str.lower()

# Remove duplicates based on 'cleaned_resume' column
df = df.drop_duplicates(subset=['cleaned_resume'])

# Save cleaned data
output_path = './data/Resume/cleaned_resume_data.csv'
df[['id', 'cleaned_resume', 'category']].to_csv(output_path, index=False)

print(f"Data cleaned and saved to {output_path}")
