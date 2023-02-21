from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def preprocess_text(text):
    
    # Remove special characters
    text = re.sub(r"[^a-zA-Z]", " ", text)

    # Convert to lowercase
    text = text.lower()
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    #Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join the tokens back into a single string
    text = " ".join(tokens)
    
    return text

# Load pre-trained model and TfidfVectorizer
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("vectorizer.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sentence = request.form['sentence']
    preprocessed_sentence = preprocess_text(sentence)
    sentence_tfidf = tfidf.transform([preprocessed_sentence])
    prediction = model.predict(sentence_tfidf)
    return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)


