import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.datasets import fetch_20newsgroups
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the 20 Newsgroups dataset
categories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
              'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles',
              'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med',
              'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast',
              'talk.politics.misc', 'talk.religion.misc']

newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
X_train_news, y_train_news = newsgroups_train.data, newsgroups_train.target

# Sample dataset of news articles and their categories
articles = [
    ("New iPhone model announced by Apple.", "Technology"),
    ("Football World Cup final match tonight.", "Sports"),
    ("Health tips for a better lifestyle.", "Health"),
    ("Stock market trends and predictions.", "Finance"),
    ("New scientific discovery in astronomy.", "Science"),
    ("Latest fashion trends for the season.", "Fashion"),
    ("Critically acclaimed drama wins Best Picture at Oscars.", "Films"),  # Added film category
    ("Action star injured on set of upcoming superhero film.", "Films")
]

# Preprocessing function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Remove non-alphanumeric characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    # Tokenize the text
    words = word_tokenize(text)
    # Remove stop words and lemmatize tokens
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Train TF-IDF vectorizer on 20 Newsgroups dataset
tfidf_vectorizer_news = TfidfVectorizer(preprocessor=preprocess_text)
X_train_tfidf_news = tfidf_vectorizer_news.fit_transform(X_train_news)

# Train the classifier on 20 Newsgroups dataset
clf_news = LinearSVC()
clf_news.fit(X_train_tfidf_news, y_train_news)

# Function to classify input text
def classify_text(input_text):
    # Preprocess input text
    input_text = preprocess_text(input_text)
    # Vectorize input text
    input_tfidf = tfidf_vectorizer_news.transform([input_text])
    # Predict the category
    prediction = clf_news.predict(input_tfidf)
    return prediction[0]

# Test the classifier with live input
while True:
    print("\nEnter your text (or 'quit' to exit):")
    input_text = input()
    if input_text.lower() == 'quit':
        break
    category_idx = classify_text(input_text)
    print("\nPredicted category (using 20 Newsgroups dataset):", categories[category_idx])

# Preprocess the text data for the sample news articles
X_train_articles, y_train_articles = zip(*articles)
X_train_preprocessed_articles = [preprocess_text(article) for article in X_train_articles]

# Train TF-IDF vectorizer on the sample news articles
tfidf_vectorizer_articles = TfidfVectorizer()
X_train_tfidf_articles = tfidf_vectorizer_articles.fit_transform(X_train_preprocessed_articles)

# Train the classifier on the sample news articles
clf_articles = LinearSVC()
clf_articles.fit(X_train_tfidf_articles, y_train_articles)

# Function to classify input text
def classify_text_articles(input_text):
    # Preprocess input text
    input_text = preprocess_text(input_text)
    # Vectorize input text
    input_tfidf = tfidf_vectorizer_articles.transform([input_text])
    # Predict the category
    prediction = clf_articles.predict(input_tfidf)
    return prediction[0]

# Test the classifier with live input
while True:
    print("\nEnter your news article (or 'quit' to exit):")
    input_text = input()
    if input_text.lower() == 'quit':
        break
    category = classify_text_articles(input_text)
    print("\nPredicted category (using sample news articles dataset):", category)
