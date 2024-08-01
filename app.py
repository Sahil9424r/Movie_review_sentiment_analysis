from flask import Flask, request, render_template
import spacy
import pickle
import numpy as np
import gensim.downloader as api

# Load the spaCy model and the GloVe model
nlp = spacy.load("en_core_web_lg")
glv = api.load("glove-wiki-gigaword-50")

# Load the trained classifier
with open('moviereview (1).pkl', 'rb') as f:
    model = pickle.load(f)

def preprocess_and_vectorize(text):
    # Remove stop words and lemmatize the text
    doc = nlp(text)
    filtered_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return glv.get_mean_vector(filtered_tokens)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    text = request.form['text']
    try:
        vector = preprocess_and_vectorize(text).reshape(1, -1)
        prediction = model.predict(vector)[0]
        label = 'positive' if prediction > 0.5 else 'negative'
        output = f'Sentiment: {label}'
        return render_template('index.html', prediction_text=output, text=text)
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}', text=text)

if __name__ == "__main__":
    app.run(debug=True)
