import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.utils import set_random_seed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

set_random_seed(41)

df = pd.read_csv('../fake_or_real_news.csv')
X = df['text']
y = df['label']

num_words = 2500
tokenizer = Tokenizer(num_words)
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)

max_sequence_length = max(len(seq) for seq in sequences)
data = pad_sequences(sequences, maxlen=max_sequence_length)

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(y)

X_non_test, X_test, y_non_test, y_test = train_test_split(data, encoded_labels, test_size = 0.3, random_state = 41)

X_train, X_val, y_train, y_val = train_test_split(X_non_test, y_non_test, test_size = 0.2, random_state = 41)

model = Sequential()
model.add(Embedding(input_dim= num_words, output_dim=100, input_length=max_sequence_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))

checkpoint_filepath = 'best_model.keras'
model_save = ModelCheckpoint(checkpoint_filepath , save_best_only=True,  monitor='val_accuracy')

model.compile(optimizer= AdamW(), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs=15, batch_size=32, callbacks = [model_save])
model.load_weights(checkpoint_filepath)
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy}')