import os
from random import randint
import re
import string

from flask import Flask, render_template, request, send_from_directory
import pandas as pd
import numpy as np
from joblib import load
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

nltk.download('stopwords')

app = Flask(__name__)
app.secret_key = os.environ.get('secret_key', str(randint(0, 1000000)))

data = pd.read_csv('./assets/cities_df', index_col=0)
X = data['Attraction']
y = data['City']

def preprocess_text(text):
    preprocessed = text.lower()
    preprocessed = re.sub('[%s]' % re.escape(string.punctuation), '', preprocessed)
    preprocessed = re.sub('\w*\d\w*','', preprocessed)
    return [preprocessed]

new_stopwords = stopwords.words('english') + list(string.punctuation)
new_stopwords += ['bali', 'barcelona', 'crete', 'dubai', 'istanbul', 'london',
                  'majorca', 'phuket', 'paris', 'rome', 'sicily', 'mallorca',
                  'goa', 'private', 'airport', 'transfer']

vectorizer = TfidfVectorizer(analyzer='word',
                             stop_words=new_stopwords,
                             decode_error='ignore')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=123)
X_train_cleaned = pd.DataFrame(X_train, columns=['Attraction'])
X_train_cleaned['cleaned'] = X_train_cleaned['Attraction'].apply(lambda x: preprocess_text(x)[0])
X_train_tfidf = vectorizer.fit_transform(X_train_cleaned['cleaned'])

model = load('./assets/non_lemmatized_model')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    input_value = request.form['input']
    preprocessed_text = preprocess_text(input_value)
    probas = model.predict_proba(vectorizer.transform(preprocessed_text))
    classes = model.classes_
    first_pred = classes[probas.argmax()]
    formatted_pred = first_pred.lower().replace(', ', '_').replace(' ', '_')
    
    city_name = formatted_pred.split('_')[0]  # Get the city name part
    image_name = f"{city_name}_wordcloud.png"  # Create the image filename

    return render_template('result.html', prediction=formatted_pred, image_name=image_name)

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(debug=True)
