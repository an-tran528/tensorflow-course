import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from os import getcwd
from tensorflow.keras.optimizers import RMSprop, Adam
import matplotlib.pyplot as plt


def get_data(filename):
    with open(filename) as training_file:
        csv_reader = csv.reader(training_file, delimiter=",")
        first_line = True
        images = []
        labels = []
        for row in csv_reader:
            if first_line:
                first_line = False
            else:
                labels.append(row[0])
                image_data = np.array_split(row[1:785], 28)
                images.append(image_data)
        images = np.array(images).astype("float")
        labels = np.array(labels).astype("float")
        # Your code ends here
    return images, labels


def train():
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        fill_mode='nearest')

    validation_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow(training_images, training_labels,
                                         batch_size=5)
    validation_generator = validation_datagen.flow(testing_images, testing_labels,
                                         batch_size=5)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation="relu",input_shape=(28,28,1)),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Conv2D(64,(3,3), activation="relu"),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(26, activation="softmax"),
    ])

    model.compile(optimizer = Adam(lr=0.0001),
                  loss = "sparse_categorical_crossentropy",
                  metrics = ["acc"])
    history = model.fit_generator(train_generator,
                        epochs=2,
                        validation_data=validation_generator,
                        verbose=1)
    model.evaluate(testing_images,testing_labels,verbose=0)

    return history


def plot(history):
    acc = history.history["acc"]
    val_acc = history.history["val_acc"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs = range(len(acc))
    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, 'r', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()


if __name__ == "__main__":
    path_sign_mnist_train = f"{getcwd()}/data/sign_mnist_train.csv"
    path_sign_mnist_test = f"{getcwd()}/data/sign_mnist_test.csv"
    training_images, training_labels = get_data(path_sign_mnist_train)
    testing_images, testing_labels = get_data(path_sign_mnist_test)
    print(training_images.shape)
    print(training_labels.shape)
    print(testing_images.shape)
    print(testing_labels.shape)

    training_images = np.expand_dims(training_images,-1)
    testing_images = np.expand_dims(testing_images,-1)
    history = train()
    plot(history)
