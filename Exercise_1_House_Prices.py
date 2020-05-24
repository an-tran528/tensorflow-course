import tensorflow as tf
import numpy as np
from tensorflow import keras


def house_model(y_new):
    xs = np.array([i for i in range(1,10000)], dtype=float)
    ys = np.array([50+50*i for i in xs], dtype=float)

    print(xs, ys)
    model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer="adam", loss="mean_squared_error")

    model.fit(xs, ys/100, epochs=100)
    return model.predict(y_new)[0]


if __name__ == "__main__":
    prediction = house_model([7.0])
    print(prediction)
