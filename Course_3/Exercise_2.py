import urllib
import csv
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from io import StringIO
import matplotlib.pyplot as plt
import io

def get_data(url):
    """Fetch csv file from url"""
    sentences = []
    labels = []
    webpage = urllib.request.urlopen(url).read().decode('utf-8')
    datafile = StringIO(webpage)
    csv_reader = csv.reader(datafile)
    next(csv_reader)
    for row in csv_reader:
        labels.append(row[0])
        sentence = row[1]
        for word in stopwords:
            token = " " + word + " "
            sentence = sentence.replace(token, " ")
            sentence = sentence.replace("  ", " ")
        sentences.append(sentence)
    print(len(sentences), len(labels))
    print(sentences[0])

    """Split data"""
    train_size = int(len(sentences)*training_portion)
    train_sentences = sentences[:train_size]
    train_labels = labels[:train_size]

    validation_sentences = sentences[train_size:]
    validation_labels = labels[train_size:]

    print(train_size, len(train_sentences), len(train_labels),
          len(validation_sentences), len(validation_labels))
    return train_sentences, train_labels, validation_sentences, validation_labels


def tokenize(train, valid,sentence = True):
    if sentence:
        tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
        tokenizer.fit_on_texts(train)
        word_index = tokenizer.word_index

        train_sequences = tokenizer.texts_to_sequences(train)
        train_padded = pad_sequences(train_sequences, padding=padding_type, maxlen=max_length)
        valid_sequences = tokenizer.texts_to_sequences(valid)
        valid_padded = pad_sequences(valid_sequences, padding=padding_type, maxlen=max_length)
        print(train_padded.shape, valid_padded.shape)
        return train_padded, valid_padded,word_index
    else:
        labels = [*train, *valid]
        label_tokenizer = Tokenizer()
        label_tokenizer.fit_on_texts(labels)
        label_word_index = label_tokenizer.word_index
        train_label_seq = np.array(label_tokenizer.texts_to_sequences(train))
        valid_label_seq = np.array(label_tokenizer.texts_to_sequences(valid))
        print(train_label_seq.shape, valid_label_seq.shape)
        return train_label_seq, valid_label_seq,label_word_index


def train(train, train_label, valid, valid_label):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24, activation="relu"),
        tf.keras.layers.Dense(6, activation="softmax"),
    ])

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    print(model.summary())

    history = model.fit(train, train_label, epochs=num_epochs,
                        validation_data=(valid, valid_label),
                        verbose = 2)
    return history, model


def plot_graph(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history["val_"+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, "val_"+string])
    plt.savefig(f"{string}.png")


def decode_sentence(text, reverse_word_index):
    return " ".join([reverse_word_index.get(i, "?") for i in text])


if __name__ == "__main__":
    url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/bbc-text.csv"

    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_portion = .8
    num_epochs = 30

    stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
            "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do",
            "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
            "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how",
            "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself",
            "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought",
            "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so",
            "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
            "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to",
            "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what",
            "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's",
            "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]

    train_sentences, train_labels, validation_sentences, validation_labels = get_data(url)
    train_padded, valid_padded, word_index = tokenize(train_sentences,validation_sentences,True)
    train_label_seq, valid_label_seq, label_word_index = tokenize(train_labels, validation_labels,False)
    history, model = train(train_padded, train_label_seq, valid_padded, valid_label_seq)
    #plot_graph(history, "loss")
    #plot_graph(history, "accuracy")

    # REVERSE WORD INDEX
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    e = model.layers[0]
    weights = e.get_weights()[0]
    print(weights.shape)

    out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
    out_m = io.open('meta.tsv', 'w', encoding='utf-8')
    for word_num in range(1, vocab_size):
        word = reverse_word_index[word_num]
        embeddings = weights[word_num]
        out_m.write(word + "\n")
        out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
    out_v.close()
    out_m.close()