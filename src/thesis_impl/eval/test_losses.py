import numpy as np

from tensorflow.keras import Sequential, layers

from eval.losses import cross_validation


def _two_bools():
    fst = np.random.randint(0, 2)
    snd = np.random.randint(0, 2)
    return fst, snd


def _xor(fst, snd):
    return int(fst != snd)


def _random_boolean_pairs(length):
    bools = np.empty((length, 2), np.int8)
    for i in range(length):
        fst, snd = _two_bools()
        bools[i] = np.array([fst, snd])
    return bools


def _xor_array(two_bools_array):
    length = len(two_bools_array)
    xors = np.empty(length, np.int8)
    for i in range(length):
        fst, snd = two_bools_array[i]
        xors[i] = _xor(fst, snd)
    return xors


def test_simulatability_loss(num_inspection_samples=200,
                             num_splits=5):

    # A simple neural network with one hidden layer.
    # We will train this model to learn a simple XOR function.
    model = Sequential()
    model.add(layers.Dense(2, input_dim=2))
    model.add(layers.Activation('tanh'))
    model.add(layers.Dense(1))
    model.add(layers.Activation('sigmoid'))

    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = _xor_array(x)

    model.compile(optimizer='Adam', loss='binary_crossentropy')
    model.fit(x, y, nb_epoch=2000)

    # A simple linear model.
    # Such a model can by definition not learn a XOR function.
    surr_model = Sequential()
    surr_model.add(layers.Dense(1, input_shape=(2,), activation='sigmoid'))
    surr_model.compile(optimizer='Adam', loss='binary_crossentropy')

    in_samples = _random_boolean_pairs(num_inspection_samples)
    out_samples = model.predict(in_samples)

    losses = cross_validation(in_samples, out_samples, surr_model,
                              num_splits=num_splits, nb_epoch=100)

    losses_sum = np.sum(losses)
    mean_loss = losses_sum / float(num_splits)

    assert mean_loss > 0.4
