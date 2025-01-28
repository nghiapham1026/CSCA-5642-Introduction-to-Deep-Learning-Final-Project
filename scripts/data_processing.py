import re
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def clean_text(text):
    """Clean text by removing HTML tags, special characters, and converting to lowercase."""
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    return text

def preprocess_data(df, max_vocab_size=10000, max_seq_length=200):
    """Clean text, tokenize, and pad sequences."""
    # Clean text
    df['cleaned_review'] = df['review'].apply(clean_text)

    # Tokenize and pad sequences
    tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['cleaned_review'])
    sequences = tokenizer.texts_to_sequences(df['cleaned_review'])
    padded_sequences = pad_sequences(sequences, maxlen=max_seq_length, padding='post', truncating='post')
    
    return padded_sequences, tokenizer

# Example usage
if __name__ == "__main__":
    df = pd.read_csv("data/raw/IMDB_Dataset.csv")
    padded_sequences, tokenizer = preprocess_data(df)
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    df.to_csv("data/processed/preprocessed_data.csv", index=False)
