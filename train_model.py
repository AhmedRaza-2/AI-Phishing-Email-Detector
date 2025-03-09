import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
df = pd.read_csv("dataset.csv")
df['Label'] = df['Email Type'].map({'Safe Email': 0, 'Phishing Email': 1})

def preprocess_text(text):
    if isinstance(text, str): 
        text = text.lower()
        text = ''.join([c for c in text if c.isalnum() or c == ' '])
    else:
        text = ''  
    return text

df['cleaned_text'] = df['Email Text'].apply(preprocess_text)
tfidf = TfidfVectorizer(max_features=1000)
X = tfidf.fit_transform(df['cleaned_text'])
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
joblib.dump(model, 'phishing_model.joblib')
joblib.dump(tfidf, 'tfidf_vectorizer.joblib')
print("Model and vectorizer saved!")