import json
import tensorflow as tf
import csv
import random
import numpy as np
import urllib
from io import StringIO
import io
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers


def get_data(url):
    """Fetch csv file from url"""
    corpus = []
    num_sentences = 0

    webpage = urllib.request.urlopen(url).read().decode('utf-8')
    datafile = StringIO(webpage)
    csv_reader = csv.reader(datafile, delimiter=",")
    for row in csv_reader:
        list_item = []
        list_item.append(row[5])
        label = row[0]
        if label == "0":
            list_item.append(0)
        else:
            list_item.append(1)
        num_sentences = num_sentences+1
        corpus.append(list_item)
    print(num_sentences)
    print(len(corpus))
    print(corpus[1])

    stn = []
    lbl = []
    random.shuffle(corpus)
    for i in range(training_size):
        stn.append(corpus[i][0])
        lbl.append(corpus[i][1])
    return stn, lbl


def tokenize(stn, lbl):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(stn)
    word_index = tokenizer.word_index
    vocab_size = len(word_index)
    sequences = tokenizer.texts_to_sequences(stn)
    padded = pad_sequences(sequences, maxlen=max_length,
                           padding=padding_type, truncating=trunc_type)
    split = int(test_portion * training_size)
    test_labels = lbl[0:split]
    test_sequences = padded[0:split]
    training_labels = lbl[split:training_size]
    training_sequences = padded[split:training_size]
    print(vocab_size, word_index["i"])
    print(len(training_sequences))
    return training_sequences, np.array(training_labels), test_sequences, np.array(test_labels), vocab_size, word_index


def embedding(url, vocab_size, word_index):
    glove = urllib.request.urlopen(url)
    embedding_index = {}
    for line in glove:
        line = line.decode("utf-8")
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embedding_index[word] = coefs

    embedding_matrix = np.zeros((vocab_size+1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    print(len(embedding_matrix))
    return embedding_matrix


def train():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size + 1, embedding_dim, input_length=max_length, weights=[embedding_matrix],
                                  trainable=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv1D(64, 5, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=4),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    num_epochs = 50
    history = model.fit(training_sequences, training_labels, epochs=num_epochs,
                        validation_data=(test_sequences, test_labels), verbose=2)

    print("Training Complete")
    return history, model


if __name__ == "__main__":
    embedding_dim = 100
    max_length = 16
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_size = 160000
    test_portion = .1

    sentences, labels = get_data(url="https://storage.googleapis.com/laurencemoroney-blog.appspot.com/training_cleaned.csv")
    training_sequences, training_labels, test_sequences, test_labels, vocab_size, word_index = tokenize(sentences, labels)

    embedding_matrix = embedding("https://storage.googleapis.com/laurencemoroney-blog.appspot.com/glove.6B.100d.txt",
                                 vocab_size, word_index)
    history, model = train()