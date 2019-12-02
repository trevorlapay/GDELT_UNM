from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding
import keras as keras

model = Sequential()

# Embedding layer
model.add(keras.layers.Flatten(input_shape=(8,)))
model.add(keras.layers.Dense(32, activation='relu'))



# Recurrent layer
model.add(LSTM(64, return_sequences=False,
               dropout=0.1, recurrent_dropout=0.1))

# Fully connected layer
model.add(Dense(64, activation='relu'))

# Dropout for regularization
model.add(Dropout(0.5))

# Output layer
model.add(Dense(21, activation='softmax'))

# Compile the model
model.compile(
    optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])