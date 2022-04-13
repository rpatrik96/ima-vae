from os import makedirs
from os.path import dirname, abspath, join, isdir

import numpy as np
import torch

from args import parse_args
from custom_generators import generate_isprites
from transforms import (
    projective_transform,
    affine_transform,
    hsv_change,
)
from utils import sprites_filename, to_one_hot

if __name__ == "__main__":
    # Command line arguments
    args = parse_args()

    nfactors = 4
    beta_params = (
        torch.Tensor(
            np.random.uniform(args.lower, args.upper, 2 * nfactors * args.nclasses)
        )
        .view(args.nclasses, nfactors, 2)
        .numpy()
    )
    angle_params = torch.zeros((args.nclasses, 2)).numpy()
    shape_probs = torch.zeros((args.nclasses, 3)).numpy()

    if args.angle:
        nfactors += 1
    if args.shape:
        nfactors += 1

    sprites_dir = join(dirname(dirname(abspath(__file__))), "sprites_data")
    if not isdir(sprites_dir):
        makedirs(sprites_dir)

    filename = sprites_filename(
        args.nobs,
        args.nclasses,
        args.projective,
        args.affine,
        args.deltah != 0 or args.deltas != 0 or args.deltav != 0,
        args.shape,
        args.angle,
        args.lower,
        args.upper,
        extension=False,
    )

    obs_per_class = int(args.nobs / args.nclasses)
    S = np.zeros((args.nclasses, obs_per_class, nfactors))
    X, Y = generate_isprites(
        args.nclasses, obs_per_class, beta_params, args, S, angle_params, shape_probs
    )
    S = torch.Tensor(S).flatten(0, 1).numpy().astype(np.float32)
    Y = to_one_hot(Y)[0].astype(np.float32)

    if args.projective is True:
        X = projective_transform(X)

    if args.affine is True:
        X = affine_transform(X)

    if args.deltah != 0 or args.deltas != 0 or args.deltav != 0:
        print("Applying color transformation in HSV space...")
        X = np.array([hsv_change(x, args.deltah, args.deltas, args.deltav) for x in X])

    np.savez_compressed(
        join(sprites_dir, filename), X, Y, S, beta_params, angle_params, shape_probs
    )
