import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Load and preprocess data
train_df = pd.read_csv('twitter_training.csv', header=None, names=['id', 'topic', 'sentiment', 'text'])
validation_df = pd.read_csv('twitter_validation.csv', header=None, names=['id', 'topic', 'sentiment', 'text'])
#combining
df = pd.concat([train_df, validation_df], ignore_index=True)

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

# Data cleaning

def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])
    tokens = word_tokenize(text)
    tokens = [ps.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)
# using apply to call the function
df['processed_text'] = df['text'].apply(preprocess_text)
borderlands_df = df[df['topic'] == 'Borderlands']

# Prepare features and labels
X = borderlands_df['processed_text']
y = borderlands_df['sentiment']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Vectorize the text
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train and evaluate Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train_vectorized, y_train)
nb_pred = nb_model.predict(X_test_vectorized)
nb_accuracy = accuracy_score(y_test, nb_pred)
nb_report = classification_report(y_test, nb_pred, output_dict=True)

# Train and evaluate Logistic Regression model
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_vectorized, y_train)
lr_pred = lr_model.predict(X_test_vectorized)
lr_accuracy = accuracy_score(y_test, lr_pred)
lr_report = classification_report(y_test, lr_pred, output_dict=True)


# Print model performance
print("\n--- Model Performance ---")
print(f"Naive Bayes Accuracy: {nb_accuracy:.4f} or {nb_accuracy*100:.2f}%")
print(f"Logistic Regression Accuracy: {lr_accuracy:.4f} or {lr_accuracy*100:.2f}% \n\n")

# Use Logistic Regression for sentiment prediction on all data
X_all_vectorized = vectorizer.transform(borderlands_df['processed_text'])
borderlands_df.loc[:, 'lr_sentiment'] = lr_model.predict(X_all_vectorized)


# Visualize sentiment distribution
fig = make_subplots(rows=1, cols=2, subplot_titles=('Original Sentiment Distribution', 'Logistic Regression Predicted Sentiment'))

for i, col in enumerate(['sentiment', 'lr_sentiment'], start=1):
    counts = borderlands_df[col].value_counts().sort_index()
    fig.add_trace(
        go.Bar(x=counts.index, y=counts.values, name=col),
        row=1, col=i
    )
    fig.update_xaxes(title_text="Sentiment", row=1, col=i)
    fig.update_yaxes(title_text="Count", row=1, col=i)

fig.update_layout(height=500, width=1000, title_text=" Sentiment Distribution Comparison")
fig.show()


# Visualize model performance
metrics = ['precision', 'recall', 'f1-score']
sentiments = ['Positive', 'Negative', 'Neutral']

fig = make_subplots(rows=1, cols=3, subplot_titles=metrics)

for i, metric in enumerate(metrics, start=1):
    nb_values = [nb_report[sentiment][metric] for sentiment in sentiments]
    lr_values = [lr_report[sentiment][metric] for sentiment in sentiments]

    fig.add_trace(
        go.Bar(x=sentiments, y=nb_values, name='Naive Bayes'),
        row=1, col=i
    )
    fig.add_trace(
        go.Bar(x=sentiments, y=lr_values, name='Logistic Regression'),
        row=1, col=i
    )

    fig.update_xaxes(title_text="Sentiment", row=1, col=i)
    fig.update_yaxes(title_text=metric.capitalize(), row=1, col=i)

fig.update_layout(height=500, width=1200, title_text="Model Performance Comparison", barmode='group')
fig.show()