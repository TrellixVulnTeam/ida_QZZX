import numpy as np

import tensorflow as tf
from sklearn.model_selection import KFold
from typeguard import typechecked


@typechecked
def cross_validation(in_samples,
                     out_samples,
                     model: tf.keras.Model,
                     num_splits: int=5,
                     **fit_kwargs):
    """
    Computes an estimate of the generalization error of `model`.
    More precisely, if one fits `model` to the mapping given by the pairs
    `(in_samples[i], out_samples[i])` and then applies the fitted function
    to other values from the same distribution as `in_samples`,
    the generalization error is the expected deviation between the
    true output value from the distribution of `out_samples`
    and the value predicted by the fitted function.

    :param in_samples:
        a sequence of values from a domain X
    :param out_samples:
        a sequence of values from a domain Y,
        such that `out_samples[i] = f(in_samples[i])`
        for all *i* and some function *f*
    :param model:
        a compiled `tf.keras.Model` with signature X -> Y
    :param num_splits:
        the number of splits to perform in the cross-validation
    :param fit_kwargs:
        optional kwargs to be passed to `model.fit()`
    :return:
        sequence of scalar loss values, each value being an estimate
        of the loss, one value per cross-validation split
    """

    cv = KFold(n_splits=num_splits, shuffle=True)

    losses = []

    for inspect_index, holdout_index in cv.split(in_samples, out_samples):
        in_inspect, out_inspect = in_samples[inspect_index], \
                                  out_samples[inspect_index]

        in_holdout, out_holdout = in_samples[holdout_index], \
                                  out_samples[holdout_index]

        model.fit(in_inspect, out_inspect, **fit_kwargs)
        loss = model.evaluate(in_holdout, out_holdout)
        losses.append(loss)

    return np.array(losses)
