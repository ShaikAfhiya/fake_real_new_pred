📰 Fake News Detection using NLP

📌 Project Overview
This project detects whether a news article is Fake News or Real News using Natural Language Processing (NLP) and Machine Learning techniques. The text data is preprocessed, converted into numerical vectors using TF-IDF Vectorization, and classified using Logistic Regression.

🎯 Objective
- Detect fake news automatically
- Perform NLP preprocessing
- Build a text classification model
- Deploy the model using FastAPI

📂 Dataset
Dataset Used:
- Fake.csv
- True.csv

Dataset contains:
- title
- text
- subject
- date

🛠 Technologies Used
Python, Pandas, NumPy, NLTK, Scikit-learn, TF-IDF, Logistic Regression, FastAPI

🔄 Workflow
1. Loaded fake and real news datasets  
2. Combined datasets and created labels  
3. Performed text preprocessing:
   - Lowercase conversion
   - Special character removal
   - Stopword removal
4. Converted text into vectors using TF-IDF  
5. Trained Logistic Regression model  
6. Evaluated using Accuracy and Recall  
7. Deployed model using FastAPI  

📊 API Example

Input
```json
{
  "text": "Government announces new healthcare policy"
}
