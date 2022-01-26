from typing import List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def get_prediction_score(mus: np.array, ys: np.array, discrete: bool) -> float:
    """ Computes the prediction score between two one dimensional, equally long arrays
    Arguments:
    mus: factors to predict
    ys: latents to predict from
    discrete: if yes, a RandomForestClassifier is used, otherwise a RandomForestRegressor
    """
    if not (len(mus.shape) == 1 and len(ys.shape) == 1 and mus.shape[0] == ys.shape[0]):
        raise ValueError("Shapes do not match.")

    model = RandomForestClassifier() if discrete is True else RandomForestRegressor()

    def split_data(mus, ys, train_split: float = 0.8, val_split: float = 0.1):
        rs = np.random.RandomState(0)
        n_datapoints = mus.shape[0]
        n_train = int(train_split * n_datapoints)
        n_train_and_val = int((train_split + val_split) * n_datapoints)
        indices = np.arange(n_datapoints)
        rs.shuffle(indices)
        mus = np.expand_dims(mus, -1)
        mus_train, ys_train = mus[:n_train], ys[:n_train]
        mus_val, ys_val = mus[n_train:n_train_and_val], ys[n_train:n_train_and_val]
        mus_test, ys_test = mus[n_train_and_val:], ys[n_train_and_val:]
        return mus_train, ys_train, mus_val, ys_val, mus_test, ys_test

    mus_train, ys_train, mus_val, ys_val, mus_test, ys_test = split_data(mus, ys)
    model.fit(mus_train, ys_train)
    ys_val_predictions, ys_test_predictions = model.predict(mus_val), model.predict(mus_test)

    def get_score(y, y_pred):
        if discrete:
            return np.mean(y == y_pred)
        else:
            score = 1 - (np.sqrt(np.mean((y - y_pred) ** 2)) / np.std(y))
            return np.clip(score, 0, 1)

    return get_score(ys_val, ys_val_predictions), get_score(ys_test, ys_test_predictions)


def mutual_information(mus: np.array, ys: np.array, discrete: List[bool]) -> Tuple[np.array]:
    """ Calculates a 'mutual information like' quantity between mus and ys

    Arguments:
    mus: mean latents
    ys: generating factors
    discrete: list specifying if corresponding dimension of ys is discrete or continuous

    Returns:
    Two matrices for val and test performance of trained classifiers/regressors.
    """
    if not ys.shape[1] == len(discrete):
        raise ValueError(f"Shapes of mus and len of discrete do not match: {mus.shape}, {len(discrete)}")
    num_codes = mus.shape[1]
    num_factors = ys.shape[1]
    m_val = np.zeros([num_codes, num_factors])
    m_test = np.zeros([num_codes, num_factors])
    for i in range(num_codes):
        for j in range(num_factors):
            m_val[i, j], m_test[i, j] = get_prediction_score(mus[:, i], ys[:, j], discrete[j])
    return m_val, m_test


def compute_mig_with_discrete_factors(mus: np.array, ys: np.array, discrete: List[bool]) -> dict:
    """ Calculates a MIG with 'mutual information like' quantity between mus and ys

        Arguments:
        mus: mean latents
        ys: generating factors
        discrete: list specifying if corresponding dimension of ys is discrete or continuous

        Returns:
        Two matrices for val and test MIG.
    """

    mi_val, mi_test = mutual_information(mus, ys, discrete)
    sorted_mi_val = np.sort(mi_val, axis=0)[::-1]
    sorted_mi_test = np.sort(mi_test, axis=0)[::-1]

    mig_val = sorted_mi_val[0, :] - sorted_mi_val[1, :]
    mig_test = sorted_mi_test[0, :] - sorted_mi_test[1, :]

    return {'mig_val': mig_val, 'mig_test': mig_test}
