from flask import Flask, request, render_template
import spacy
import pickle
import numpy as np

# Load the spaCy model and the trained classifier
nlp = spacy.load('en_core_web_lg')
with open('emailspam.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    # Extract data from form
    text = request.form['text']
    
    try:
        # Process the input text
        doc = nlp(text)
        vector = doc.vector.reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(vector)
        label = 'spam' if prediction[0] == 1 else 'ham'
        
        # Prepare the output
        output = f'Classification: {label}'
        
        return render_template('index.html', prediction_text=output, text=text)
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}', text=text)

if __name__ == "__main__":
    app.run(debug=True)
