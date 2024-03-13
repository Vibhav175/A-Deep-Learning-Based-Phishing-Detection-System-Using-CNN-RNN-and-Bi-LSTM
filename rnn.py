from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

def build_rnn_model(max_words, max_len):
    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=50, input_length=max_len))
    model.add(LSTM(units=64, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2, activation='softmax'))  # Assuming two classes: benign and phishing

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def train_rnn_model(model, X_train, y_train, X_test, y_test, epochs=5, batch_size=32):
    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

    # Evaluate the model on test data
    accuracy = model.evaluate(X_test, y_test)[1]
    print(f"Accuracy on test data: {accuracy}")

    return model