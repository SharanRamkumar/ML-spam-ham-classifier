# 📩 ML-Spam-Ham Classifier

![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-purple?logo=pandas)
![Status](https://img.shields.io/badge/Status-Completed-success)
![License](https://img.shields.io/badge/License-MIT-green)

A machine learning project that classifies SMS messages as **Spam** or **Ham (legitimate)** using **TF-IDF feature extraction** and **Logistic Regression**.

The project covers the complete classical machine learning workflow, including data cleaning, feature engineering, model training, validation, threshold selection, evaluation, error analysis, and prediction on unseen messages.

---

## 🎯 Project Objective

The goal of this project is to build a binary text classification model capable of determining whether an SMS message is:

- 🟢 **Ham** — legitimate message
- 🔴 **Spam** — unwanted or fraudulent message

The project was also developed to understand the mathematical and practical foundations behind a traditional machine learning classification pipeline.

---

## 🧠 Machine Learning Pipeline

```text
Raw SMS Dataset
       ↓
Data Cleaning
       ↓
Duplicate Removal
       ↓
Train / Validation / Test Split
       ↓
TF-IDF Feature Extraction
       ↓
Logistic Regression
       ↓
Probability Prediction
       ↓
Threshold Selection
       ↓
Final Evaluation
       ↓
Error Analysis
       ↓
Prediction on New Messages
```

---

## 📊 Dataset

The project uses the **SMS Spam Collection** dataset.

The original dataset contains 5,572 SMS messages. Duplicate messages were removed during preprocessing, resulting in approximately **5,169 unique messages** used for this project.

Each record contains:

```text
label    message
```

where the label is either:

```text
ham
spam
```

### Dataset Split

The cleaned dataset was divided into:

| Dataset | Samples |
|---|---:|
| Training | 3,308 |
| Validation | 827 |
| Testing | 1,034 |
| **Total** | **5,169** |

The split was stratified to preserve the proportion of ham and spam messages.

---

# 🧹 Data Preprocessing

Before training the model, duplicate messages were removed to reduce the possibility of the same or highly similar examples appearing multiple times in the dataset.

The dataset was then separated into:

```text
X → message text
y → spam/ham label
```

The data was split into training, validation, and test sets.

The test set was kept separate for final evaluation.

---

# 🔢 TF-IDF Feature Extraction

Machine learning algorithms cannot directly process raw text.

Therefore, SMS messages were converted into numerical vectors using **TF-IDF (Term Frequency–Inverse Document Frequency)**.

The basic idea is to assign higher importance to words that are:

- frequent within a particular message
- relatively uncommon across the overall dataset

The TF-IDF value can be represented as:

$$
TFIDF(t,d) = TF(t,d) \times IDF(t)
$$

where:

- \(t\) = term/word
- \(d\) = document/message
- \(TF(t,d)\) = frequency of the term in the document
- \(IDF(t)\) = inverse document frequency

A simplified IDF formulation is:

$$
IDF(t) = \log\left(\frac{N}{df(t)}\right)
$$

where:

- \(N\) = total number of documents
- \(df(t)\) = number of documents containing the term

The fitted TF-IDF vectorizer was used to transform the training, validation, and test data.

The vectorizer was fitted **only on the training data** to prevent information from the validation or test sets from influencing feature construction.

---

# 🤖 Logistic Regression

The numerical TF-IDF vectors were used as input to a **Logistic Regression** classifier.

For a binary classification problem, Logistic Regression calculates:

$$
z = w^Tx + b
$$

where:

- \(x\) = input feature vector
- \(w\) = learned model weights
- \(b\) = bias/intercept

The result is passed through the sigmoid function:

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

This produces a probability between 0 and 1.

For example:

```text
P(spam) = 0.82
```

means the model estimates an 82% probability that the message is spam.

---

# 📉 Loss Function

Logistic Regression is trained using **log loss / binary cross-entropy**.

For a single example:

$$
L = -[y\log(p)+(1-y)\log(1-p)]
$$

where:

- \(y\) = actual class
- \(p\) = predicted probability

The loss becomes larger when the model is confidently wrong.

For example:

```text
Actual = spam
Prediction = 0.95 spam
→ low loss
```

while:

```text
Actual = spam
Prediction = 0.05 spam
→ high loss
```

The training process adjusts the model's weights to minimize this loss.

---

# 🎚️ Probability Threshold

Instead of automatically using a 0.50 threshold, different thresholds were evaluated using the validation set.

The decision rule is:

$$
P(spam) \geq threshold \Rightarrow spam
$$

The evaluated thresholds included:

| Threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.20 | 0.91 | 0.93 | 0.92 |
| **0.25** | **0.95** | **0.91** | **0.93** |
| 0.30 | 0.93 | 0.89 | 0.93 |
| 0.35 | 0.98 | 0.83 | 0.90 |
| 0.40 | 0.99 | 0.78 | 0.87 |
| 0.45 | 0.99 | 0.77 | 0.87 |
| 0.50 | 0.99 | 0.75 | 0.85 |

A threshold of **0.25** was selected because it provided the best F1 score among the evaluated thresholds.

---

# 📈 Final Model Performance

The final model was evaluated on the held-out test set using the selected threshold of 0.25.

### Confusion Matrix

```text
                Predicted
                Ham    Spam

Actual Ham      897     6
Actual Spam      2     129
```

### Classification Report

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Ham | 1.00 | 0.99 | 1.00 |
| Spam | 0.96 | 0.98 | 0.97 |
| **Overall Accuracy** | | | **0.99** |

### Key Metrics

- **Accuracy:** ~99%
- **Spam Precision:** 96%
- **Spam Recall:** 98%
- **Spam F1-score:** 97%

---

# 🔍 Error Analysis

The model's incorrect predictions were analyzed rather than relying solely on accuracy.

## False Negatives

False negatives are messages that are actually spam but were classified as ham.

Examples included messages containing unusual vocabulary or context-dependent spam patterns.

This demonstrates a limitation of TF-IDF + Logistic Regression: the model does not truly understand semantic context.

---

## False Positives

False positives are legitimate messages that were classified as spam.

Some legitimate messages contained words such as:

```text
call
request
text
reply
```

which can also occur frequently in spam messages.

The model therefore learned statistical associations between vocabulary and spam rather than understanding the complete meaning of the message.

---

# 🔬 Model Interpretation

One advantage of Logistic Regression is that its learned coefficients can be inspected.

Words with strongly positive coefficients are associated with the spam class.

Examples from the trained model included:

```text
call
txt
free
text
stop
mobile
reply
claim
uk
www
prize
won
```

Words with negative coefficients were more strongly associated with ham messages.

This makes Logistic Regression relatively interpretable compared with many more complex machine learning models.

---

# 🧪 Prediction on New Messages

The trained model and TF-IDF vectorizer were saved using `joblib`.

The prediction pipeline is:

```text
New SMS
   ↓
Saved TF-IDF Vectorizer
   ↓
Numerical Feature Vector
   ↓
Saved Logistic Regression Model
   ↓
Spam Probability
   ↓
Threshold = 0.25
   ↓
HAM / SPAM
```

Example:

```text
Input:
"Congratulations! You have won a free prize. Call now!"

Output:
Prediction: SPAM
Spam probability: ...
```

The prediction script can be executed using:

```bash
python src/predict_mail.py
```

---

# 📁 Project Structure

```text
sms-spam-classifier/
│
├── data/
│   └── raw/
│       └── SMSSpamCollection
│
├── models/
│   ├── spam_classifier.pkl
│   └── vectorizer.pkl
│
├── src/
│   ├── load_data.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict_mail.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

# 🛠️ Technologies Used

- **Python**
- **Pandas**
- **Scikit-learn**
- **TF-IDF**
- **Logistic Regression**
- **Joblib**
- **Git / GitHub**

---

# 🚀 How to Run

## 1. Clone the repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd sms-spam-classifier
```

## 2. Create a virtual environment

```bash
python -m venv .venv
```

## 3. Activate the environment

### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

## 5. Train the model

```bash
python src/train.py
```

## 6. Evaluate the model

```bash
python src/evaluate.py
```

## 7. Predict a new message

```bash
python src/predict_mail.py
```

---

# ⚠️ Limitations

This project uses a classical machine learning approach and has several limitations:

1. The model relies heavily on word-level statistical patterns.
2. TF-IDF does not capture deep semantic meaning.
3. Unseen vocabulary can reduce prediction quality.
4. Context and word order are only partially represented.
5. The dataset consists of SMS messages, so performance may not directly generalize to modern email datasets.
6. The dataset may not represent current spam patterns.

---

# 🔮 Future Improvements

Possible future improvements include:

- Testing on a larger and more recent spam dataset
- Comparing Logistic Regression with Naive Bayes and other classifiers
- Character-level TF-IDF
- N-gram features
- Hyperparameter optimization
- Calibration of predicted probabilities
- Transformer-based text classification
- Deployment through a web API
- Real-time spam classification

---

# 🎓 Learning Outcomes

Through this project, I gained practical experience with:

- Data cleaning and duplicate removal
- Feature engineering
- Text vectorization
- TF-IDF mathematics
- Logistic Regression
- Sigmoid functions
- Log loss / binary cross-entropy
- Train/validation/test methodology
- Probability-based classification
- Decision threshold optimization
- Precision, recall and F1-score
- Confusion matrix analysis
- Error analysis
- Model interpretation
- Saving and loading trained ML models
- Making predictions on unseen data

---

## 📌 Conclusion

This project demonstrates a complete classical machine learning workflow for text classification, from raw data preprocessing to model training, evaluation, interpretation, and prediction on unseen messages.

The final Logistic Regression classifier achieved approximately **99% accuracy** on the held-out test set, with an **97% F1-score for the spam class** using a validation-selected threshold of 0.25.
