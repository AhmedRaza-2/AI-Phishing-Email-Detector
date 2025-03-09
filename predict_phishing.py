import joblib
import re

model = joblib.load('phishing_model.joblib')
tfidf = joblib.load('tfidf_vectorizer.joblib')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  
    text = re.sub(r'[^\w\s]', '', text)  
    return text.strip()

def predict_email(email_text):
    cleaned_email = clean_text(email_text)  
    email_tfidf = tfidf.transform([cleaned_email])  
    

    if email_tfidf.nnz == 0:  
        return "WARNING: The input text contains no known words from the training set!"
    
    prediction = model.predict(email_tfidf)  
    return "PHISHING EMAIL DETECTED! " if prediction[0] == 1 else "EMAIL IS SAFE"

if __name__ == "__main__":
    print("\nPhishing Email Detector (Type 'exit' to quit)")
    while True:
        email_text = input("\nEnter an email to check:\n")
        if email_text.lower() == 'exit':
            print("Goodbye!")
            break
        
        result = predict_email(email_text)
        print(f"\nResult: {result}\n")
