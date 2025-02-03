<<<<<<< HEAD
import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('pickle.pkl', 'rb'))
    
@app.route('/')
def home():
    return render_template('index.html')
   

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        
        LIMIT_BAL=int(request.form["LIMIT_BAL"])
        PAY_0=int(request.form["PAY_0"])
        PAY_2=int(request.form["PAY_2"])
        PAY_3=int(request.form["PAY_3"])
        PAY_4=int(request.form["PAY_4"])
        PAY_5=int(request.form["PAY_5"])
        PAY_6=int(request.form["PAY_6"])
        PAY_AMT1=int(request.form["PAY_AMT1"])
        PAY_AMT2=int(request.form["PAY_AMT2"])
        PAY_AMT3=int(request.form["PAY_AMT3"])
        PAY_AMT6=int(request.form["PAY_AMT6"])
        
        
                      
        data = np.array([[LIMIT_BAL,PAY_0,PAY_2,PAY_3,PAY_4,PAY_5,PAY_6,PAY_AMT1,PAY_AMT2,PAY_AMT3,PAY_AMT6]])
        my_prediction = model.predict(data)
        
        return render_template('result.html', prediction=my_prediction)
              
if __name__ == '__main__':
    app.run(host="localhost", port=2000)
=======
from flask import Flask, request, render_template
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# Load the model and vectorizer
model = joblib.load('comment_analysis_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Initialize Flask application
app = Flask(__name__)
    
def preprocess_text(text):
    
    # Remove non-alphabetic characters
    text = "".join([char for char in text if char.isalpha() or char.isspace()])

    # Remove extra whitespace
    text = text.strip()

    # Load Bengali stopwords
    stop_words = stopwords.words("bengali")
    
    # Tokenize the text and remove stop words
    text = " ".join([word for word in word_tokenize(text) if word not in stop_words])
    
    return text

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        comment = request.form['comment']
        processed_comment = preprocess_text(comment)
        features = vectorizer.transform([processed_comment])
        prediction = model.predict(features)
        return render_template('index.html', prediction=prediction[0], comment=comment)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True,port= 3000)
>>>>>>> 119ff98e (Initial commit - Credit Card Default Prediction)
