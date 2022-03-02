from typing import Tuple

import numpy as np
from ima_vae.metrics.mig import mutual_information


def test_mutual_information():
    ys, mus, mixing = generate_test_data(1000)
    m_val, m_test = mutual_information(mus, ys, [1, 0, 0])
    print(mixing)
    print(m_val)
    print(m_test)
    assert np.all(np.argmax(m_val, axis=0) == np.argmax(mixing, axis=0))


def generate_test_data(number_of_samples: int) -> Tuple[np.array]:
    """Generates a test dataset with one discrete and two continuous
    generating factors, mapped to a five dimensional latent
    """
    np.random.seed(0)
    discrete = np.random.randint(0, 5, (number_of_samples, 1))
    continuous = np.random.rand(number_of_samples, 2) * 4
    generating_factors = np.concatenate((discrete, continuous), axis=1)
    mixing = np.array(
        [[0.1, 0.1, 0.1], [0.1, 0.1, 1], [0.1, 1, 0.1], [1, 0.1, 0.1], [1, 1, 1]]
    )
    latents = generating_factors @ mixing.T
    return generating_factors, latents, mixing
