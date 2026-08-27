import joblib

vectorizer=joblib.load("vectorizer.pkl")
model=joblib.load("spam_classifier.pkl")

message=input("Enter a message:")
message_tfidf=vectorizer.transform([message])
spam_probabilities=model.predict_proba(message_tfidf)[0][1]

threshold=0.25
prediction="spam" if spam_probabilities>=threshold else "ham"
print(f"spam probability: {spam_probabilities:.2%}")
print(f"Prediction:{prediction.upper()}")

