from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense

def build_cnn_model(max_words, max_len):
    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=50, input_length=max_len))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(2, activation='softmax'))  # Assuming two classes: benign and phishing
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_cnn_model(model, X_train, y_train, X_test, y_test, epochs=5, batch_size=32):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
    accuracy = model.evaluate(X_test, y_test)[1]
    print(f"Accuracy on test data: {accuracy}")

    return model