# train_model.py
"""
Script to train a simple Sentiment Analysis model using Logistic Regression and TF-IDF Vectorizer.
Generates:
  - tfidf_vectorizer.pkl
  - sentiment_model.pkl
This provides the Flask application with pre-trained models for immediate execution.
"""

import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 1. Define sample dataset for Positive, Negative, and Neutral reviews
reviews = [
    # Positive Reviews
    ("This product is absolutely amazing! Highly recommended.", "Positive"),
    ("Great value for money. The quality is outstanding.", "Positive"),
    ("I love this product. It works perfectly and is very durable.", "Positive"),
    ("Excellent customer service and very fast shipping.", "Positive"),
    ("Best purchase I have made in a long time. 5 stars!", "Positive"),
    ("Very happy with the purchase. Fits perfectly and looks great.", "Positive"),
    ("Superb build quality. Exceeded my expectations.", "Positive"),
    ("Works like a charm. Very easy to set up and use.", "Positive"),
    ("Fantastic item! Will buy again.", "Positive"),
    ("Highly impressed with the performance. Solid product.", "Positive"),
    ("This is good.", "Positive"),
    ("This is great.", "Positive"),
    ("This is very good.", "Positive"),
    ("Excellent product, I really like it.", "Positive"),
    ("I love it, works perfectly.", "Positive"),
    ("Perfect item, very fast delivery and amazing quality.", "Positive"),
    ("Highly recommended! Perfect.", "Positive"),
    ("Wonderful experience, very satisfied.", "Positive"),
    ("Superb, brilliant, awesome!", "Positive"),
    ("So happy with this purchase.", "Positive"),
    
    # Negative Reviews
    ("Terrible product, it broke on the first day of use.", "Negative"),
    ("Worst experience ever. Do not waste your money on this.", "Negative"),
    ("Very disappointed with the quality. It feels cheap.", "Negative"),
    ("The item arrived damaged and the customer support was useless.", "Negative"),
    ("Does not work at all. Completely useless and defective.", "Negative"),
    ("Poor quality materials. It started falling apart after a week.", "Negative"),
    ("Waste of money and time. Very low quality.", "Negative"),
    ("Horrible performance. It constant crashes and fails.", "Negative"),
    ("Do not buy this. It is a complete scam and doesn't match description.", "Negative"),
    ("Extremely unsatisfied. It is way too expensive for what it offers.", "Negative"),
    ("This is bad.", "Negative"),
    ("This is terrible.", "Negative"),
    ("This is very bad.", "Negative"),
    ("Worst product ever, I hate it.", "Negative"),
    ("It broke immediately, poor quality.", "Negative"),
    ("Horrible customer service and extremely slow shipping.", "Negative"),
    ("Do not buy! Waste of money.", "Negative"),
    ("Completely broken and useless.", "Negative"),
    ("Very bad quality, cheap build.", "Negative"),
    ("Extremely disappointed.", "Negative"),
    
    # Neutral Reviews
    ("It is an okay product, nothing special but does the job.", "Neutral"),
    ("Average quality, it is decent for the price.", "Neutral"),
    ("Neutral feeling about this. It works, but has some flaws.", "Neutral"),
    ("It is acceptable, but there are better options available.", "Neutral"),
    ("Not bad, but not great either. Just average.", "Neutral"),
    ("It works as advertised, but the build is just ordinary.", "Neutral"),
    ("Just fine. Nothing to write home about, but not terrible.", "Neutral"),
    ("Decent performance, but I expected slightly more features.", "Neutral"),
    ("An ordinary item. Neither good nor bad.", "Neutral"),
    ("It is middle of the road. Standard product.", "Neutral"),
    ("This is okay.", "Neutral"),
    ("It is average.", "Neutral"),
    ("Not good, not bad.", "Neutral"),
    ("It works fine.", "Neutral"),
    ("Nothing special, just ordinary.", "Neutral"),
    ("Satisfactory but could be better.", "Neutral"),
    ("It is acceptable.", "Neutral"),
    ("Mediocre quality.", "Neutral"),
    ("Just an average item.", "Neutral"),
    ("Decent but not excellent.", "Neutral")
]

# 2. Split dataset into text and labels
X_train = [review[0] for review in reviews]
y_train = [review[1] for review in reviews]

def train_and_save_models():
    print("Initializing vectorizer and fitting text...")
    # Initialize the TF-IDF Vectorizer with word level ngrams to capture negatives like "not good"
    vectorizer = TfidfVectorizer(lowercase=True, stop_words=None, ngram_range=(1, 2))
    X_train_vectorized = vectorizer.fit_transform(X_train)
    
    print("Training Logistic Regression model...")
    # Initialize and train Logistic Regression model
    model = LogisticRegression(C=5.0, max_iter=1000)
    model.fit(X_train_vectorized, y_train)
    
    # Save the vectorizer
    vectorizer_path = 'tfidf_vectorizer.pkl'
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"Saved TF-IDF Vectorizer to: {vectorizer_path}")
    
    # Save the model
    model_path = 'sentiment_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Saved Sentiment Model to: {model_path}")
    
    print("Training completed successfully!")

if __name__ == "__main__":
    train_and_save_models()
