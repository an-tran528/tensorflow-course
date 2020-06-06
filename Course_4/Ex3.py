import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def plot_series(time, series, string, format="-", start=0, end=None ):
    print("SAVING", string)
    plt.figure(figsize=(10, 6))
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(False)
    plt.savefig(f"{string}.png")


def trend(time, slope=0):
    return slope * time


def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.1,
                    np.cos(season_time * 6 * np.pi),
                    2 / np.exp(9 * season_time))


def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


def train(data, window_size):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Lambda(lambda x: tf.expand_dims(x,axis=-1),
                               input_shape=[None]),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(1),
        tf.keras.layers.Lambda(lambda x: x*100)
    ])
    """
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: 1e-8 * 10**(epoch/20)
    )
    """
    model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9), metrics=["mae"])
    history = model.fit(dataset, epochs=100, verbose=2)

    return model,history


def forecasting(model, series, window_size, split_time):
    forecast = []
    for time in range(len(series) - window_size):
        forecast.append(model.predict(series[time:time+window_size][np.newaxis]))
    forecast = forecast[split_time-window_size:]
    return np.array(forecast)[:, 0, 0]


if __name__ == "__main__":
    tf.keras.backend.clear_session()
    tf.random.set_seed(51)
    np.random.seed(51)

    time = np.arange(10*365+1, dtype="float32")
    baseline= 10
    series = trend(time, 0.1)
    amplitude = 40
    slope = 0.005
    noise_level = 3
    split_time = 3000

    time = np.arange(10 * 365 + 1, dtype="float32")
    baseline = 10
    series = trend(time, 0.1)
    baseline = 10
    amplitude = 40
    slope = 0.005
    noise_level = 3

    # Create the series
    series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
    # Update with noise
    series += noise(time, noise_level, seed=51)
    plot_series(time, series, "Original series")

    split_time = 3000
    time_train = time[:split_time]
    x_train = series[:split_time]
    time_valid = time[split_time:]
    x_valid = series[split_time:]

    window_size = 20
    batch_size = 32
    shuffle_buffer_size = 1000
    tf.keras.backend.clear_session()
    dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
    model, history = train(dataset, window_size)
    results = forecasting(model, series, window_size, split_time)

    plot_series(time_valid, x_valid, "valid")
    plot_series(time_valid, results, "result")

    print(tf.keras.metrics.mean_absolute_error(x_valid, results).numpy())