from flask import Flask, render_template, request
import string
import re
import pickle
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load the vectorizer (TF-IDF vectorizer)
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load the trained model (SVC model)
model = joblib.load("svc_model.pkl")

# Initialize Flask app
app = Flask(__name__)

# Function for text cleaning
def text_cleaning(text):
    """Cleans text by removing punctuation, numbers, new lines, and links."""
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = text.translate(str.maketrans('', '', '0123456789'))  # Remove numbers
    text = re.sub('\n', '', text)  # Remove new lines
    return text

# Function for text preprocessing
def preprocess_text(text):
    tokens = word_tokenize(text)  # Tokenization
    stop_words = set(stopwords.words('english')) - set(["not"])  # Stopwords (keeping "not" for negation)
    filtered_tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return " ".join(filtered_tokens)

# Function to predict sentiment
def predict_sentiment(review):
    """Predicts sentiment of a review and returns emoji-based output."""
    review = text_cleaning(review)
    review = preprocess_text(review)
    review_vectorized = vectorizer.transform([review])
    review_output = model.predict(review_vectorized)  # Use the correct model variable

    if review_output == [1]:
        return "POSITIVE 😊", "green"
    elif review_output == [2]:
        return "NEUTRAL 😐", "gray"
    else:
        return "NEGATIVE 😡", "red"

# Define the homepage route
@app.route("/", methods=["GET", "POST"])
def home():
    sentiment = ""
    color = "gray"  # Default color
    review = ""

    if request.method == "POST":
        review = request.form["review"]
        sentiment, color = predict_sentiment(review)

    return render_template("index.html", review=review, sentiment=sentiment, color=color)

if __name__ == "__main__":
    app.run(debug=True)
