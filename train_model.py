import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer # it used to covert data to digits
from sklearn.ensemble import RandomForestClassifier  #prebuilt libray that is used to as tree decision maker or like step by step
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib #store the model 
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
tfidf = TfidfVectorizer(max_features=1000) #extract the email features based on priotity

X = tfidf.fit_transform(df['cleaned_text']) 
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100)

model.fit(X_train, y_train)     #here now x is having vectorized filtered data and y has 0 or 1 as phishing or not 
y_pred = model.predict(X_test)

print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
#saving models back
joblib.dump(model, 'phishing_model.joblib')
joblib.dump(tfidf, 'tfidf_vectorizer.joblib')
print("Model and vectorizer saved!")
