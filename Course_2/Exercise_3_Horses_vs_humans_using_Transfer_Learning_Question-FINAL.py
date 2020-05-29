# ATTENTION: Please do not alter any of the provided code in the exercise. Only add your own code where indicated
# ATTENTION: Please do not add or remove any cells in the exercise. The grader will check specific cells based on the cell position.
# ATTENTION: Please use the provided epoch values when training.

# Import all the necessary files!
import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from os import getcwd
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import zipfile
import shutil
import matplotlib.pyplot as plt


def train_inceptionv3():
    class myCallBacks(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if logs.get("val_acc") > 0.97:
                self.model.stop_training = True

    path_inception = f"{getcwd()}/../data/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"

    # Create an instance of the inception model from the local pre-trained weights
    local_weights_file = path_inception

    pre_trained_model = InceptionV3(input_shape=(150, 150, 3),
                                    include_top=False,
                                    weights=None)

    pre_trained_model.load_weights(local_weights_file)

    # Make all the layers in the pre-trained model non-trainable
    for layer in pre_trained_model.layers:
        # Your Code Here
        layer.trainable = False

    last_layer = pre_trained_model.get_layer("mixed7")
    print('last layer output shape: ', last_layer.output_shape)
    last_output = last_layer.output
    # Flatten the output layer to 1 dimension
    x = layers.Flatten()(last_output)
    # Add a fully connected layer with 1,024 hidden units and ReLU activation
    x = layers.Dense(1024, activation="relu")(x)
    # Add a dropout rate of 0.2
    x = layers.Dropout(0.2)(x)
    # Add a final sigmoid layer for classification
    x = layers.Dense (1, activation="sigmoid")(x)

    model = Model(pre_trained_model.input, x)

    model.compile(optimizer = RMSprop(lr=0.0001),
                  loss = "binary_crossentropy",
                  metrics = ["acc"])

    # Add our data-augmentation parameters to ImageDataGenerator
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

    # Note that the validation data should not be augmented!
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Flow training images in batches of 20 using train_datagen generator
    train_generator = train_datagen.flow_from_directory(train_dir,
                                                       batch_size=10,
                                                        class_mode="binary",
                                                        target_size=(150,150))


    # Flow validation images in batches of 20 using test_datagen generator
    validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                            batch_size=10,
                                                            class_mode="binary",
                                                            target_size=(150,150))

    callbacks = myCallBacks()
    history = model.fit_generator(train_generator,
                                  epochs=3,
                                  validation_data=validation_generator,
                                  verbose=1,
                                  callbacks=[callbacks])
    return history

def plot_accuracy(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend(loc=0)
    plt.figure()

    plt.show()


if __name__ = "__main__":
    path_horse_or_human = f"{getcwd()}/data/horse-or-human.zip"
    # Get the Horse or Human Validation dataset
    path_validation_horse_or_human = f"{getcwd()}/data/validation-horse-or-human.zip"
    #shutil.rmtree('/data')
    local_zip = path_horse_or_human
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('/data/training')
    zip_ref.close()

    local_zip = path_validation_horse_or_human
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('/data/validation')
    zip_ref.close()

    train_dir = '/data/training'
    validation_dir = '/data/validation'

    train_horses_dir = train_dir + "/horses"
    train_humans_dir = train_dir + "/humans"
    validation_horses_dir = validation_dir + "/horses"
    validation_humans_dir = validation_dir + "/humans"

    train_horses_fnames = os.listdir(train_horses_dir)
    train_humans_fnames = os.listdir(train_humans_dir)
    validation_horses_fnames = os.listdir(validation_horses_dir)
    validation_humans_fnames = os.listdir(validation_humans_dir)

    print(len(train_horses_fnames))
    print(len(train_humans_fnames))
    print(len(validation_horses_fnames))
    print(len(validation_humans_fnames))

    history = train_inceptionv3()
    plot_accuracy(history)