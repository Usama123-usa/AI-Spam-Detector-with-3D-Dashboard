# train_and_save_model.py
import pandas as pd
import numpy as np
import re
import string
import pickle  # For saving/loading model components

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load CSV
# Make sure 'spam.csv' is in the same directory as this script
try:
    df = pd.read_csv('spam.csv', encoding='latin-1')
except FileNotFoundError:
    print("Error: 'spam.csv' file not found. Please make sure the file is in the correct directory.")
    # If file is not found, create a dummy DataFrame for demonstration
    # (This section runs only if the actual file is missing)
    data = {'Category': ['ham', 'spam', 'ham', 'spam', 'ham'],
            'Message': ['Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat?',
                        'URGENT! You have won a 1 week FREE membership in our $100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010 T&C www.dbuk.net LMBTCMORPN. 118',
                        'Nah I dont think he goes to usf, he is a private account.',
                        'Had your mobile 11 months or more? U R entitled To Update to the latest colour mobiles with camera for FREE! Call The Mobile Update Co FREE on 08002986030',
                        'I HAVE A DATE ON SUNDAY WITH WILL!!']}
    df = pd.DataFrame(data)
    print("Using dummy dataset because 'spam.csv' was not found.")

df = df[['Category', 'Message']]  # Keep only necessary columns
df.columns = ['label', 'text']  # Rename columns
df.dropna(inplace=True)  # Remove missing values

# Map labels to numeric: 'ham' = 0, 'spam' = 1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

print("Initial part of the DataFrame (head):")
print(df.head())
print("\nLabel distribution:")
print(df['label'].value_counts())  # Shows label distribution

# Text cleaning function
def clean_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Remove links
    text = re.sub(r'\@w+|\#','', text)  # Remove mentions and hashtags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    return text

df['clean_text'] = df['text'].apply(clean_text)  # Apply cleaning function
print("\nExample of cleaned text:")
print(df[['text', 'clean_text']].head())

# TF-IDF Vectorizer
# Removes 'english' stop words and limits to top 3000 features
tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
X_tfidf = tfidf.fit_transform(df['clean_text']).toarray()  # Convert text to TF-IDF features
y = df['label'].values  # Convert labels to numpy array

# Standardize TF-IDF data
# with_mean=False is required for sparse matrices
scaler = StandardScaler(with_mean=False)
X_scaled = scaler.fit_transform(X_tfidf)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Logistic Regression model using GridSearchCV
param_grid = {
    'C': [0.01, 0.1, 1, 10],  # Regularization strength
    'penalty': ['l1', 'l2'],  # Regularization type
    'solver': ['liblinear']   # 'liblinear' supports both 'l1' and 'l2'
}

# Set up GridSearchCV
# max_iter=1000 increases iterations for convergence
# n_jobs=-1 uses all available CPU cores
grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, verbose=1, n_jobs=-1)
print("\nStarting GridSearchCV...")
grid.fit(X_train, y_train)

best_model = grid.best_estimator_  # Get the best model
print(f"\n📌 Best Parameters: {grid.best_params_}")

y_pred = best_model.predict(X_test)  # Make predictions on test data
print("✅ Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")  # Print accuracy
print("\nClassification Report:\n", classification_report(y_test, y_pred))  # Print classification report

# Save the trained components
print("\nSaving model components...")
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)  # Save TF-IDF vectorizer
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)  # Save scaler
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)  # Save best model
print("Model components saved: tfidf_vectorizer.pkl, scaler.pkl, best_model.pkl")
