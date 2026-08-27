import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

df=pd.read_csv("data/raw/SMSSpamCollection",sep="\t",header=None,names=["label","message"]")
X=df["message"]
y=df["label"]
df=df.drop_duplicates()
X_train,X_temp,y_train,y_temp=train_test_split(X,y,test_size=0.36,random_state=42)
X_val,X_test,y_val,y_test=train_test_split(X_temp,y_temp,test_size=0.5556,random_state=42)

vectorizer=TfidfVectorizer()
X_train_tfidf=vectorizer.fit_transform(X_train)
X_val_tfidf=vectorizer.transform(X_val)
X_test_tfidf=vectorizer.transform(X_test)

model=LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf,y_train)

joblib.dump(vectorizer,"vectorizer.pkl")
joblib.dump(model,"spam_classifier.pkl")
