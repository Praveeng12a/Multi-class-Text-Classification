# Multi-class-Text-Classification
Resume Classification Application

This project is a multi-class text classification application for identifying the category of resumes based on their descriptions. It uses data from the Resume Dataset.

Table of Contents

Description

Environment Setup

File Execution Workflow

Output

Description

The application processes resume data, trains a multi-class text classification model, and provides a user-friendly interface to predict the category of resumes. The dataset used for this project contains various categories of resumes and is sourced from Kaggle: Resume Dataset.

Environment Setup

Create a Virtual Environment

Use Conda to create a Python 3.10 environment for this project:

conda create --name resume_classification python=3.10
conda activate resume_classification

Install Dependencies

Install the required Python packages from requirements.txt:

pip install -r requirements.txt

File Execution Workflow

1. Data Cleaning

Run the clean_resume.py script to clean and preprocess the input data.

python3 clean_resume.py

2. Data Preprocessing

Run the preprocess_data.py script to prepare the cleaned data for model training.

python3 preprocess_data.py

3. Model Training

Train the multi-class classification model using the train_model.py script.

python3 train_model.py

4. Application Deployment

Launch the application with the deploy_app.py script to start the Streamlit interface for inference.

streamlit run deploy_app.py

Output

Below is an example of the application interface showing the resume classification output:



Hosting and Deployment

This application is hosted using GitHub and Cloud Connect for seamless access and deployment.

Contributing

Feel free to fork this repository and contribute to enhancing the application by submitting pull requests or reporting issues.

License

This project is licensed under the MIT License. See the LICENSE file for details.
