import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Load CSV files with error handling
try:
    fake_news_data = pd.read_csv('Fake.csv', error_bad_lines=False, engine='python')
    true_news_data = pd.read_csv('True.csv', error_bad_lines=False, engine='python')
except pd.errors.ParserError as e:
    print(f"Error reading CSV files: {e}")
    fake_news_data = pd.DataFrame()  # Create empty DataFrames to prevent errors
    true_news_data = pd.DataFrame()

# Add labels to distinguish between fake and true news
fake_news_data['label'] = 1
true_news_data['label'] = 0

# Combine the datasets into one
combined_data = pd.concat([fake_news_data, true_news_data], ignore_index=True)
combined_data = combined_data.sample(frac=1).reset_index(drop=True)

# Split the data into training, validation, and test sets
X = combined_data['text']
y = combined_data['label']
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Tokenize and preprocess text data
max_words = 10000  
max_len = 400     
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_val = tokenizer.texts_to_sequences(X_val)
X_test = tokenizer.texts_to_sequences(X_test)

X_train = pad_sequences(X_train, maxlen=max_len)
X_val = pad_sequences(X_val, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

# Define and train the LSTM model
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=128, input_length=max_len))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(64))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Train the model
batch_size = 32
epochs = 10

history = model.fit(X_train, y_train, 
                    batch_size=batch_size, epochs=epochs, 
                    validation_data=(X_val, y_val))

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f'Test accuracy: {test_accuracy * 100:.2f}%')
