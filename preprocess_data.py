import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def encode_labels(df, target_column):
    label_encoder = LabelEncoder()
    df["label_code"] = label_encoder.fit_transform(df[target_column])
    return df

def tokenize_and_pad_sequences(texts, max_words):
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    max_len = len(sequences[0])
    padded_sequences = pad_sequences(sequences, maxlen=max_len)
    return padded_sequences, max_len

def preprocess_data(file_path, target_column, max_words):
    df = load_data(file_path)
    df = encode_labels(df, target_column)
    X_sequences, max_len = tokenize_and_pad_sequences(df['url'], max_words)
    return X_sequences, to_categorical(df['label_code']), max_len
