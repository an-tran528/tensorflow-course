import os
import zipfile
import random
import tensorflow as tf
import shutil
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
from os import getcwd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    files = []
    for f in os.listdir(SOURCE):
        files.append(SOURCE+f)
    train_sample = random.sample(files, int(SPLIT_SIZE*len(files)))
    test_sample = [f for f in files if f not in train_sample]
    for f in train_sample:
        filename = f.split("/")[-1]
        copyfile(f,TRAINING+filename)
    for f in test_sample:
        filename = f.split("/")[-1]
        copyfile(f,TESTING+filename)


def train_cats_dogs():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3,3), activation="relu",input_shape=(150,150,3)),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Conv2D(32,(3,3), activation="relu"),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(1,activation="sigmoid")
    ])

    model.compile(optimizer=RMSprop(lr=0.001),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        batch_size=10,
                                                        class_mode="binary",
                                                        target_size=(150,150))
    validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                      batch_size=10,
                                                      class_mode="binary",
                                                      target_size=(150,150))
    history = model.fit_generator(train_generator,
                                  epochs=5,
                                  verbose=1,
                                  validation_data=validation_generator)
    return history

def plot_acc(history):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs = range(len(acc))

    plt.plot(epochs,acc, "r", "Training accuracy")
    plt.plot(epochs,val_acc, "r", "Validation accuracy")
    plt.title('Training and validation accuracy')
    plt.figure()

    plt.plot(epochs,loss, "r", "Training loss")
    plt.plot(epochs,val_loss, "r", "Validation loss")
    plt.title('Training and validation loss')


if __name__ == "__main__":
    path_cats_and_dogs = f"{getcwd()}/data/cats-and-dogs.zip"
    shutil.rmtree(f"{getcwd()}/data/cats-v-dogs")
    local_zip = path_cats_and_dogs
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall(f"{getcwd()}/data")
    zip_ref.close()
    print(len(os.listdir(f"{getcwd()}/data/PetImages/Cat/")))
    print(len(os.listdir(f"{getcwd()}/data/PetImages/Dog/")))

    try:
        # YOUR CODE GOES HERE
        base_dir = f"{getcwd()}/data/cats-v-dogs"

        train_dir = os.path.join(base_dir, 'training')
        validation_dir = os.path.join(base_dir, 'testing')

        # Directory with our training cat/dog pictures
        train_cats_dir = os.path.join(train_dir, 'cats')
        train_dogs_dir = os.path.join(train_dir, 'dogs')

        # Directory with our validation cat/dog pictures
        validation_cats_dir = os.path.join(validation_dir, 'cats')
        validation_dogs_dir = os.path.join(validation_dir, 'dogs')

        os.mkdir(base_dir)
        os.mkdir(train_dir)
        os.mkdir(validation_dir)

        os.mkdir(train_cats_dir)
        os.mkdir(train_dogs_dir)

        os.mkdir(validation_cats_dir)
        print(validation_cats_dir)
        os.mkdir(validation_dogs_dir)
    except OSError:
        pass

    CAT_SOURCE_DIR = f"{getcwd()}/data/PetImages/Cat/"
    TRAINING_CATS_DIR = f"{getcwd()}/data/cats-v-dogs/training/cats/"
    TESTING_CATS_DIR = f"{getcwd()}/data/cats-v-dogs/testing/cats/"
    DOG_SOURCE_DIR = f"{getcwd()}/data/PetImages/Dog/"
    TRAINING_DOGS_DIR = f"{getcwd()}/data/cats-v-dogs/training/dogs/"
    TESTING_DOGS_DIR = f"{getcwd()}/data/cats-v-dogs/testing/dogs/"

    split_size = .9
    split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
    split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)

    history = train_cats_dogs()