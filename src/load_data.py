import pandas as pd
from sklearn.model_selection import train_test_split


# Load dataset
df = pd.read_csv("data/processed/processed_data.csv)

print("Original dataset shape:", df.shape)

# Check class distribution
print("\nClass distribution:")
print(df["label"].value_counts())

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Check duplicates
print("\nDuplicate rows:", df.duplicated().sum())

# Remove duplicate messages
df = df.drop_duplicates()

print("\nRows after removing duplicates:", len(df))
print("Duplicates remaining:", df.duplicated().sum())


# Separate features and labels
X = df["message"]
y = df["label"]


# Split into train and test sets
X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Split remaining data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.2,
    random_state=42,
    stratify=y_temp
)

print("\nDataset split:")
print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Testing samples:", len(X_test))
