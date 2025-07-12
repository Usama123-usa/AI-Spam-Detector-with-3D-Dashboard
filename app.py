# app.py
from flask import Flask, request, jsonify, render_template  # render_template is imported
from flask_cors import CORS
import pickle
import re
import string
import os

app = Flask(__name__)
CORS(app)

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text

# Load trained model components
try:
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('best_model.pkl', 'rb') as f:
        best_model = pickle.load(f)
    print("Model components loaded successfully.")
except FileNotFoundError:
    print("Error: Model components (tfidf_vectorizer.pkl, scaler.pkl, best_model.pkl) not found.")
    print("Please run 'train_and_save_model.py' first to generate these files.")
    exit()
except Exception as e:
    print(f"Error loading model components: {e}")
    exit()

# New Route: Serve index.html on root URL ('/')
@app.route('/')
def index():
    # Flask automatically searches for files in the 'templates' folder
    return render_template('index.html')

# Prediction API endpoint (same as before)
@app.route('/predict', methods=['POST'])
def predict():
    if not request.is_json:
        return jsonify({"error": "Request must be in JSON format"}), 400

    data = request.get_json()
    email_text = data.get('emailText')

    if not email_text:
        return jsonify({"error": "No emailText provided"}), 400

    try:
        cleaned_text = clean_text(email_text)
        tfidf_vec = tfidf.transform([cleaned_text])
        tfidf_vec_scaled = scaler.transform(tfidf_vec)

        prediction = best_model.predict(tfidf_vec_scaled)[0]
        prediction_label = "spam" if prediction == 1 else "not-spam"

        return jsonify({"prediction": prediction_label})

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
