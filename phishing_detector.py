 # Import Libraries
import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------------------------------
# Sample Dataset
# ---------------------------------------------------

data = {
    'email': [
        'Congratulations! You won a free lottery. Click here now.',
        'Your bank account has been suspended. Verify immediately.',
        'Meeting scheduled for tomorrow regarding project updates.',
        'Please review the attached report before the meeting.',
        'Win cash prizes now!!! Visit this link urgently.',
        'Lunch at 1 PM today?',
        'Update your password immediately to avoid suspension.',
        'Project submission deadline extended to next week.'
    ],
    
    'label': [
        'Phishing',
        'Phishing',
        'Safe',
        'Safe',
        'Phishing',
        'Safe',
        'Phishing',
        'Safe'
    ]
}

df = pd.DataFrame(data)

# ---------------------------------------------------
# Data Preprocessing
# ---------------------------------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

df['clean_email'] = df['email'].apply(clean_text)

# ---------------------------------------------------
# Feature Extraction using TF-IDF
# ---------------------------------------------------

vectorizer = TfidfVectorizer(stop_words='english')

X = vectorizer.fit_transform(df['clean_email'])
y = df['label']

# ---------------------------------------------------
# Split Dataset
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)

# ---------------------------------------------------
# Train Model
# ---------------------------------------------------

model = MultinomialNB()

model.fit(X_train, y_train)

# ---------------------------------------------------
# Predictions
# ---------------------------------------------------

y_pred = model.predict(X_test)

# ---------------------------------------------------
# Evaluation
# ---------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print("\\nClassification Report:\\n")
print(classification_report(y_test, y_pred))

# ---------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Phishing', 'Safe'],
            yticklabels=['Phishing', 'Safe'])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ---------------------------------------------------
# Test Custom Email
# ---------------------------------------------------

sample_email = ["Urgent! Your account will be blocked. Click now."]

sample_clean = [clean_text(email) for email in sample_email]

sample_vector = vectorizer.transform(sample_clean)

prediction = model.predict(sample_vector)

print("\\nCustom Email Prediction:", prediction[0])