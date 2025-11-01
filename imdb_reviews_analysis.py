# %%
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, SpatialDropout1D, Embedding

# Load dataset
df = pd.read_csv(r"IMDB Dataset.csv")  # ENTER .csv file location here
print("Dataset shape:", df.shape)
print(df.head())

# Optional: drop duplicates or nulls
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Encode sentiment labels
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
sentiment_label = df['sentiment'].values
texts = df['review'].values

# Tokenization
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(texts)
vocab_size = len(tokenizer.word_index) + 1

encoded_docs = tokenizer.texts_to_sequences(texts)
padded_sequence = pad_sequences(encoded_docs, maxlen=200)

# Build the LSTM model
model = Sequential()
model.add(Embedding(vocab_size, 32, input_length=200))
model.add(SpatialDropout1D(0.25))
model.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Summary
model.summary()

# Train the model
history = model.fit(padded_sequence, sentiment_label, validation_split=0.2, epochs=5, batch_size=32)
print("Model training complete!")

# Accuracy Plot
plt.plot(history.history['accuracy'], label='acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.title("Accuracy over Epochs")
plt.show()

# Loss Plot
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.title("Loss over Epochs")
plt.show()

# Sentiment prediction function
def predict_sentiment(text):
    tw = tokenizer.texts_to_sequences([text])
    tw = pad_sequences(tw, maxlen=200)
    prediction = int(model.predict(tw).round().item())
    print("Predicted label:", "positive" if prediction == 1 else "negative")

# %%

# Run predictions when script is executed directly
if __name__ == "__main__":
    predict_sentiment("I loved this movie! It was fantastic.")
    predict_sentiment("This was the worst film I've ever watched.")



