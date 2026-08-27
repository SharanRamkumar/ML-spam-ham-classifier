import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

vectorizer = joblib.load("vectorizer.pkl")
model = joblib.load("spam_classifier.pkl")

df = pd.read_csv("data/raw/SMSSpamCollection",sep="\t",header=None,names=["label","message"]")

X = df["message"]
y = df["label"]

X_train, X_temp, y_train, y_temp = train_test_split(X,y,test_size=0.36,random_state=42,stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp,y_temp,test_size=0.5556,random_state=42,stratify=y_temp)
X_test_tfidf = vectorizer.transform(X_test)

test_probabilities = model.predict_proba(X_test_tfidf)
test_spam_probabilities = test_probabilities[:, 1]

threshold = 0.25
final_predictions = [
    "spam" if probability >= threshold else "ham"
    for probability in test_spam_probabilities
]

print("Final Confusion Matrix:")
print(confusion_matrix(y_test, final_predictions))

print("\nFinal Classification Report:")
print(classification_report(y_test, final_predictions))


#  Error Analysis — False Negatives
false_negatives = X_test[(y_test == "spam") & (pd.Series(final_predictions, index=y_test.index) == "ham")]

print("\nFalse Negatives:")
print(false_negatives)


#  Error Analysis — False Positives
false_positives = X_test[(y_test == "ham") & (pd.Series(final_predictions, index=y_test.index) == "spam"]

print("\nFalse Positives:")
print(false_positives)
