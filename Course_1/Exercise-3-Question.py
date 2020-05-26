import tensorflow as tf
from os import path, getcwd, chdir


def train_mnist_conv():

    class myCallBacks(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs = {}):
            if logs.get("accuracy") >= 0.998:
                print("\nReached 99.8% accuracy so cancelling training!")
                self.model.stop_training = True

    mnist = tf.keras.datasets.mnist

    (training_images, training_labels), (test_images, test_labels) = mnist.load_data(path=path)
    training_images = training_images.reshape(training_images.shape[0], training_images.shape[1], training_images.shape[2],1)

    test_images = test_images.reshape(test_images.shape[0], test_images.shape[1], test_images.shape[2], 1)

    training_images, test_images = training_images/255.0, test_images/255.0

    callbacks = myCallBacks()
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation=tf.nn.relu, input_shape=(28,28,1)),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    history = model.fit(training_images, training_labels, epochs=10, callbacks=[callbacks])
    return history.epoch, history.history["accuracy"][-1]


if __name__ == "__main__":
    path = f"{getcwd()}/data/mnist.npz"
    """  
    tf_1x
    config = tf.ConfigProto()
    config.gpu.options.allow_growth = True
    sess = tf.Session(config=config)
    """
    epoch, accuracy = train_mnist_conv()
    print("EPOCH", epoch, "ACC",accuracy)



