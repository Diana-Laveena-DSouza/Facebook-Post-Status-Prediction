from flask import Flask, request, render_template
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import TaggedDocument
import nltk
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")
nltk.download('omw-1.4')

app = Flask(__name__)

# Load the Models
doc2vec = pickle.load(open('models/doc2vec_model.pkl', 'rb'))
model_classifier = pickle.load(open('models/svc_model.pkl', 'rb'))
stsc_model = pickle.load(open('models/stsc_model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
# Function for Prediction
def predict():
    text_data = [x for x in request.form.values()]

    # Text Preprocessing Steps

    # Lower the text
    text_lower = [text.lower() for text in text_data]

    # Remove hyperlink
    text_hyper_rem = [re.sub(r'http.*.com', ' ', text) for text in text_lower]

    # Remove @name
    text_at_rem = [re.sub(r'@[a-z]+', ' ', text) for text in text_hyper_rem]

    # Remove Special Character
    text_spl_rem = [re.sub(r'[^a-zA-Z]+', ' ', text) for text in text_at_rem]

    # Word Tokenize
    text_token = [word_tokenize(text) for text in text_spl_rem]

    # Remove Stop Words and Stem
    stop_words = list(set(stopwords.words('english')) - {'not', 'no'})
    lemm = WordNetLemmatizer()
    text_lem = []
    for text in text_token:
        text_lem.append([lemm.lemmatize(word) for word in text if word not in stop_words])
    # Create test documents
    test_document = [TaggedDocument(words=x, tags=[i]) for i, x in enumerate(text_lem)]
    print(test_document)
    # Doc2Vec Model
    test_vec = doc2vec.infer_vector(test_document[0].words).reshape(1, 3500)

    # Feature Scaling
    test_final = stsc_model.transform(test_vec)

    # Prediction
    pred = model_classifier.predict(test_final)

    if pred == 0:
        prediction = 'Anger'
    elif pred == 1:
        prediction = 'Fear'
    elif pred == 2:
        prediction = 'Joy'
    elif pred == 3:
        prediction = 'Love'
    elif pred == 4:
        prediction = 'Sadness'
    else:
       prediction = 'Surprise'
    return render_template('index.html', post_text=text_data[0], prediction_text = prediction)


if __name__ == "__main__":
    app.run(debug = True)
