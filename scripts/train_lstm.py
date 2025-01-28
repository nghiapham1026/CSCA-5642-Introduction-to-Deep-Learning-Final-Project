from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from sklearn.model_selection import train_test_split
import pandas as pd

def build_lstm_model(vocab_size, embedding_dim, input_length, lstm_units, dropout_rate):
    """Build and compile an LSTM model."""
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length),
        LSTM(lstm_units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(64, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Example usage
if __name__ == "__main__":
    df = pd.read_csv("data/processed/preprocessed_data.csv")
    X_train, X_test, y_train, y_test = train_test_split(df['cleaned_review'], df['sentiment'], test_size=0.5, random_state=42)

    lstm_model = build_lstm_model(vocab_size=10000, embedding_dim=128, input_length=200, lstm_units=128, dropout_rate=0.2)
    lstm_model.fit(X_train, y_train, validation_split=0.2, epochs=5, batch_size=32, verbose=2)
    lstm_model.save("models/lstm_model.h5")
