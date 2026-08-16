import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import os

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

RAW_PATH = "/content/CCHVAE/yelpzip.csv"

# ✅ Force proper splitting: YelpZip is comma-separated with quotes
df = pd.read_csv(
    RAW_PATH,
    sep=",",              # explicit comma delimiter
    quotechar='"',        # handle quoted text correctly
    encoding="utf-8",
    on_bad_lines="skip",
    engine="python"
)

print("Columns detected:", df.columns)
print("Sample rows:\n", df.head())

# ✅ Assign expected column names if needed
expected_cols = ["review_id","user_id","prod_id","rating","label","date","text","tag"]
if len(df.columns) == len(expected_cols):
    df.columns = expected_cols

# ✅ Map labels to binary (YelpZip: -1 = fake, 1 = real)
if "label" in df.columns:
    df['label'] = df['label'].map({-1:1, 1:0})
else:
    raise ValueError("❌ 'label' column not found. Check delimiter/columns in your dataset.")

# ✅ Clean text
df['clean_text'] = df['text'].apply(clean_text)

# ✅ Keep only useful columns
df = df[['clean_text', 'label']].dropna()

print("Label distribution:\n", df['label'].value_counts())

# ✅ Train/Validation/Test split
train_df, temp_df = train_test_split(df, test_size=0.3,
                                     stratify=df['label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5,
                                   stratify=temp_df['label'], random_state=42)

print("Train size:", len(train_df))
print("Validation size:", len(val_df))
print("Test size:", len(test_df))

# ✅ Save outputs
os.makedirs("/content/CCHVAE/processed", exist_ok=True)
train_df.to_csv("/content/CCHVAE/processed/train_preprocessed.csv", index=False)
val_df.to_csv("/content/CCHVAE/processed/val_preprocessed.csv", index=False)
test_df.to_csv("/content/CCHVAE/processed/test_preprocessed.csv", index=False)

print("✅ Preprocessing complete. Files saved in /content/CCHVAE/processed/")
