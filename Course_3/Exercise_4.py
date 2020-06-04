from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import tensorflow.keras.utils as ku
import numpy as np
import urllib


def tokenize(corpus):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index)+1

    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            input_sequences.append(token_list[:i+1])
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences,
                                              maxlen=max_sequence_len,
                                              padding="pre"))
    predictors, label = input_sequences[:,:-1], input_sequences[:,-1]
    print(label)
    label = ku.to_categorical(label, num_classes=total_words)
    print(label.shape, total_words)
    return predictors, label,max_sequence_len, total_words, tokenizer


def train(predictors, labels):
    model = Sequential()
    model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
    model.add(Bidirectional(LSTM(150, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(LSTM(100))
    model.add(Dense(total_words/2, activation="relu",
                    kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(total_words, activation="softmax"))
    model.compile(loss="categorical_crossentropy",
                  optimizer=Adam(lr=0.001),
                  metrics=["accuracy"])
    print(model.summary())
    history = model.fit(predictors, labels, epochs = 100, verbose=1)
    return history, model

def test (seed_text, tokenizer, model):
    next_words = 100

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text


if __name__ == "__main__":
    url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sonnets.txt"
    sonnets = urllib.request.urlopen(url).read().decode("utf-8")
    corpus = sonnets.lower().split("\n")
    predictors, labels ,max_sequence_len, total_words, tokenizer = tokenize(corpus)
    history, model = train(predictors,labels)
    seed_text = "Help me Obi Wan Kenobi, you're my only hope"
    generated_text = test(seed_text, tokenizer, model)
    print(generated_text)