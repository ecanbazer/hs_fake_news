import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder



df = pd.read_csv('../fake_or_real_news.csv')
X = df['text']
y = df['label']

num_words = 2500
tokenizer = Tokenizer(num_words)
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)

# Step 3: Pad the sequences to have the same length
max_sequence_length = max(len(seq) for seq in sequences)
data = pad_sequences(sequences, maxlen=max_sequence_length)

# Step 4: Convert labels to numerical values
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(data, encoded_labels, test_size = 0.3, random_state = 41)

# Step 6: Build the model
model = Sequential()
model.add(Embedding(input_dim= num_words, output_dim=32, input_length=max_sequence_length))
model.add(LSTM(32, dropout = 0.2))
model.add(Dense(1, activation='sigmoid'))

checkpoint_filepath = 'best_model.hdf5'
model_save = ModelCheckpoint(checkpoint_filepath , save_best_only=True,  monitor='val_accuracy')

# Step 7: Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 8: Train the model
model.fit(x_train, y_train, validation_split = 0.2, epochs=15, batch_size=32, callbacks = [model_save])
model.load_weights(checkpoint_filepath)
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test loss: {loss}')
print(f'Test accuracy: {accuracy}')