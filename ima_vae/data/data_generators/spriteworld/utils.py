import numpy as np


def to_one_hot(x, m=None):
    "batch one hot"
    if type(x) is not list:
        x = [x]
    if m is None:
        ml = []
        for xi in x:
            ml += [xi.max() + 1]
        m = max(ml)
    dtp = x[0].dtype
    xoh = []
    for i, xi in enumerate(x):
        xoh += [np.zeros((xi.size, int(m)), dtype=dtp)]
        xoh[i][np.arange(xi.size), xi.astype(np.int)] = 1
    return xoh


def sprites_filename(
    n_obs,
    n_classes,
    projective: bool = False,
    affine: bool = False,
    hsv_change: bool = False,
    shape: bool = False,
    angle: bool = False,
    lower: int = 2,
    upper: int = 15,
    extension: bool = True,
):

    filename = (
        "isprites_nclasses_"
        + str(n_classes)
        + "_nobs_"
        + str(int(n_obs))
        + "_lower_"
        + str(lower)
        + "_upper_"
        + str(upper)
    )

    if angle is True:
        filename += "_angle"
    if shape is True:
        filename += "_shape"
    if projective is True:
        filename += "_projective"
    if affine is True:
        filename += "_affine"
    if hsv_change is True:
        filename += "_deltahsv"

    if extension is True:
        filename += ".npz"

    return filename
