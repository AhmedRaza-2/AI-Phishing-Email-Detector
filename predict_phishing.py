import joblib
import re
import tkinter as tk
from tkinter import scrolledtext, messagebox
from nltk.stem import PorterStemmer

# Loading trained model and vectorized file 
model = joblib.load('phishing_model.joblib')
tfidf = joblib.load('tfidf_vectorizer.joblib')

stemmer = PorterStemmer()

def clean_text(text):
    text = text.lower()  # to convert data to lower case 
    text = re.sub(r'[^\w\s]', ' ', text)  # it will remove special characters
    text = ' '.join([stemmer.stem(word) for word in text.split()])  # convertw ord from running -> run 
    return text.strip()

# main fucntion to predict email 
def predict_email():
    email_text = email_input.get("1.0", tk.END).strip() 
    
    if not email_text:
        messagebox.showwarning("Input Error", "Please enter an email to analyze!")
        return

    cleaned_email = clean_text(email_text)  
    email_tfidf = tfidf.transform([cleaned_email])  

    if email_tfidf.nnz == 0:  
        result = "WARNING: No known words from training set! Email may be too short."
    else:
        prediction = model.predict(email_tfidf)  
        confidence = model.predict_proba(email_tfidf)[0][1]  

        if prediction[0] == 1:
            result = f"PHISHING EMAIL DETECTED! (Accuracy: {confidence:.2f})"
        else:
            result = f"EMAIL IS SAFE (Accuracy: {1 - confidence:.2f})"
   
    result_label.config(text=result)

# to create ui 
root = tk.Tk()
root.title("Phishing Email Detector")
root.geometry("500x400")
tk.Label(root, text="Phishing Email Detector", font=("Arial", 16, "bold")).pack(pady=10)
email_input = scrolledtext.ScrolledText(root, width=60, height=10)
email_input.pack(pady=10)

#button
predict_btn = tk.Button(root, text="Check Email", command=predict_email, font=("Arial", 12), bg="blue", fg="white")
predict_btn.pack(pady=5)

#output area
result_label = tk.Label(root, text="", font=("Arial", 12, "bold"), wraplength=400)
result_label.pack(pady=10)

root.mainloop()
