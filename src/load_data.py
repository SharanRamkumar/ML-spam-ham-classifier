import pandas as pd
import re
from pathlib import Path


# File paths
RAW_PATH = Path("data/raw/SMSSpamCollection")
PROCESSED_PATH = Path("data/processed/processed_data.csv")


# Load raw dataset
df = pd.read_csv(RAW_PATH,sep="\t",header=None,names=["label", "message"])
print("Original dataset shape:", df.shape)

# Remove missing values
df = df.dropna(subset=["label", "message"])

# Remove duplicate messages
df = df.drop_duplicates(subset=["message"])

# Text cleaning function
def clean_text(text):

    # Convert to lowercase
    text = text.lower()

    # Replace URLs
    text = re.sub(r"https?://\S+|www\.\S+"," URL ",text)

    # Replace email addresses
    text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b"," EMAIL ",text)
    
    # Keep letters, numbers, whitespace, and currency symbols
    text = re.sub(r"[^a-z0-9\s£$€₹]", " ", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# Create cleaned message
df["clean_message"] = df["message"].apply(clean_text)
df = df.dropna(subset=["clean_message"])
# Replace missing values with empty strings
df["clean_message"] = df["clean_message"].fillna("")

# Remove messages that contain no letters
df = df[df["clean_message"].str.strip() != ""]


# Keep only the three fields we need
df = df[ ["label",  "clean_message"] ]

# Create processed directory if needed
PROCESSED_PATH.parent.mkdir(parents=True,exist_ok=True)

# Save processed dataset
df.to_csv(PROCESSED_PATH,index=False)

# Display results
print("\nProcessed dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nSample:")
print(df.head())

print(f"\nProcessed data saved to: {PROCESSED_PATH}")
