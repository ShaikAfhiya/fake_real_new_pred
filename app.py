from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import re
from nltk.corpus import stopwords
# Load model and vectorizer
with open('fake_real_pkl.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer_pkl.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
# FastAPI app
app = FastAPI()
# stopwords
import nltk
nltk.download('stopwords')
stop_words = stopwords.words('english')
# preprocessing function
def formatted_text(text):
    clean_text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = clean_text.lower().split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Input schema
class NewsData(BaseModel):
    text: str

# Prediction API
@app.post('/')
def predict(data: NewsData):

    # preprocess input text
    clean_news = formatted_text(data.text)

    # vectorize
    vector_input = vectorizer.transform([clean_news])

    # prediction
    prediction = model.predict(vector_input)[0]

    if prediction == 0:
        result = "Fake News"
    else:
        result = "Real News"

    return {
        "Prediction": result
    }
