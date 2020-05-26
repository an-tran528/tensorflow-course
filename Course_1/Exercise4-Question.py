import tensorflow as tf
import os
import zipfile
from os import path, getcwd, chdir
from tensorflow.keras.optimizers import RMSprop
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def train_happy_sad_model():
    DESIRED_ACCURACY = 0.999

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if logs.get("accuracy") > DESIRED_ACCURACY:
                print("\nReached 99.9% accuracy so cancelling training")
                self.model.stop_training = True

    callbacks = myCallback()

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3,3), activation="relu", input_shape=(150,150,3)),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Conv2D(32, (3,3), activation="relu"),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(loss="binary_crossentropy",
                  optimizer =RMSprop(lr=0.001),
                  metrics = ["accuracy"],
                  )

    train_datagen = ImageDataGenerator(rescale=1 / 255)
    #validation_datagen = ImageDataGenerator(rescale=1/255)

    train_generator = train_datagen.flow_from_directory(
        f"{getcwd()}/data/h-or-s/",
        target_size=(150,150),
        batch_size=16,
        class_mode="binary",
    )
    """
    validation_datagen = validation_datagen.flow_from_directory(
        "/data/h-or-s/",
        target_size=(300,300),
        batch_size=128,
        class_mode="binary",
    )
    """
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=5,
        epochs=15,
        verbose=1,
        callbacks=[callbacks]
    )

    return history.history["accuracy"]


if __name__ == "__main__":
    path = f"{getcwd()}/data/happy-or-sad.zip"
    zip_ref = zipfile.ZipFile(path, "r")
    zip_ref.extractall(f"{getcwd()}/data/h-or-s")
    zip_ref.close()
    train_happy_sad_model()
