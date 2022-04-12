import argparse
from os import makedirs
from os.path import dirname, abspath, join, isdir

import cv2
import numpy as np
import torch

from ima_vae.data.utils import to_one_hot
from spriteworld import environment as spriteworld_environment
from spriteworld import factor_distributions as distribs
from spriteworld import renderers as spriteworld_renderers
from spriteworld import sprite_generators
from spriteworld import tasks


def random_sprites_config(beta_params, label):
    factor_list = [
        distribs.Beta("x", beta_params[label][0][0], beta_params[label][0][1]),
        distribs.Beta("y", beta_params[label][1][0], beta_params[label][1][1]),
        distribs.Beta("scale", beta_params[label][2][0], beta_params[label][2][1]),
        # We are using HSV, so "c0 = H", "c1 = S", "c2 = V"
        distribs.Beta("c0", beta_params[label][3][0], beta_params[label][3][1]),
        distribs.Continuous("c1", 1.0, 1.0),
        distribs.Continuous("c2", 1.0, 1.0),
    ]

    if args.angle:
        angles = np.random.uniform(args.lower, args.upper, 2)
        angle_params[label] = angles
        factor_list.append(
            distribs.Beta("angle", angle_params[label][0], angle_params[label][1])
        )

    if args.shape:
        probs = np.random.uniform(0, 1, 3)
        probs = probs / probs.sum()
        shape_probs[label] = probs
        factor_list.append(
            distribs.Discrete(
                "shape", ["triangle", "square", "pentagon"], probs=shape_probs[label]
            )
        )
    else:
        factor_list.append(distribs.Discrete("shape", ["triangle"]))

    factors = distribs.Product(factor_list)
    sprite_gen = sprite_generators.generate_sprites(factors, num_sprites=1)

    renderers = {
        "image": spriteworld_renderers.PILRenderer(
            image_size=(64, 64),
            anti_aliasing=5,
            color_to_rgb=spriteworld_renderers.color_maps.hsv_to_rgb,
        ),
        "attributes": spriteworld_renderers.SpriteFactors(
            factors=("x", "y", "shape", "angle", "scale", "c0", "c1", "c2")
        ),
    }

    config = {
        "task": tasks.NoReward(),
        "action_space": None,
        "renderers": renderers,
        "init_sprites": sprite_gen,
        "max_episode_length": 1,
    }
    return config


def collect_frames(config, label, num_frames):
    """Instantiate config as environment and get single images from it."""
    env = spriteworld_environment.Environment(**config)
    images = []
    for i in range(num_frames):
        ts = env.reset()
        S[label, i, 0] = env._sprites[0].x[0]
        S[label, i, 1] = env._sprites[0].y[0]
        S[label, i, 2] = env._sprites[0].scale[0]
        S[label, i, 3] = env._sprites[0].c0[0]
        if args.angle:
            S[label, i, 4] = env._sprites[0].angle[0]
        if args.shape:
            if env._sprites[0].shape == "triangle":
                S[label, i, 5] = 0
            elif env._sprites[0].shape == "square":
                S[label, i, 5] = 1
            elif env._sprites[0].shape == "pentagon":
                S[label, i, 5] = 2

        images.append(ts.observation["image"])
    return images


def generate_isprites(num_classes, obs_per_class, lower, upper):
    for i in range(num_classes):
        print(i)
        if i == 0:
            full_obs = collect_frames(
                random_sprites_config(beta_params, i), i, obs_per_class
            )
            full_labels = np.zeros(obs_per_class)
        else:
            full_obs += collect_frames(
                random_sprites_config(beta_params, i), i, obs_per_class
            )
            full_labels = np.concatenate((full_labels, np.ones(obs_per_class) * i))

    return np.array(full_obs), np.array(full_labels)


def hsv_change(
    img: np.ndarray, delta_h: int = 0, delta_s: int = 0, delta_v: int = 0
) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    h = cv2.add(h, delta_h)
    s = cv2.add(s, delta_s)
    v = cv2.add(v, delta_v)

    hsv = cv2.merge([h, s, v])
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return rgb


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nclasses", type=int, default=1, help="Number of auxiliary variables"
    )
    parser.add_argument(
        "--nobs", type=int, default=10000, help="Number of observations in dataset"
    )
    parser.add_argument(
        "--lower",
        type=int,
        default=2,
        help="Lower bound on alpha and beta (Set to at least 2)",
    )
    parser.add_argument(
        "--upper", type=int, default=15, help="Upper bound on alpha and beta"
    )
    parser.add_argument(
        "--angle", type=bool, default=False, help="True if you want angle as a factor"
    )
    parser.add_argument(
        "--shape", type=bool, default=False, help="True if you want shape as a factor"
    )
    parser.add_argument(
        "--project",
        type=bool,
        default=False,
        help="True if you want to apply a projective transformation (to destroy colum-ortogonality)",
    )
    parser.add_argument(
        "--affine",
        type=bool,
        default=False,
        help="True if you want to apply an affine transformation (to destroy colum-ortogonality)",
    )
    parser.add_argument(
        "--deltah", type=int, default=0, help="Disturbance in the Hue channel"
    )
    parser.add_argument(
        "--deltas", type=int, default=0, help="Disturbance in the Saturation channel"
    )
    parser.add_argument(
        "--deltav", type=int, default=0, help="Disturbance in the Value channel"
    )
    args = parser.parse_args()

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

    sprites_dir = join(dirname(dirname(abspath(__file__))), "sprites_data")
    if not isdir(sprites_dir):
        makedirs(sprites_dir)

    filename = (
        "isprites_"
        + "nclasses_"
        + str(args.nclasses)
        + "_nobs_"
        + str(args.nobs)
        + "_lower_"
        + str(args.lower)
        + "_upper_"
        + str(args.upper)
    )

    if args.angle:
        nfactors += 1
        filename += "_angle"
    if args.shape:
        nfactors += 1
        filename += "_shape"
    if args.project:
        filename += "_projective"
    if args.affine:
        filename += "_affine"
    if args.deltah != 0 or args.deltas != 0 or args.deltav != 0:
        filename += "_deltahsv"

    obs_per_class = int(args.nobs / args.nclasses)
    S = np.zeros((args.nclasses, obs_per_class, nfactors))
    X, Y = generate_isprites(args.nclasses, obs_per_class, args.lower, args.upper)
    S = torch.Tensor(S).flatten(0, 1).numpy().astype(np.float32)
    Y = to_one_hot(Y)[0].astype(np.float32)

    if args.project is True:

        print("Applying projective transformation...")
        rows, cols = X.shape[1], X.shape[2]
        src_points = np.float32(
            [[0, 0], [0, rows - 1], [cols / 2, 0], [cols / 2, rows - 1]]
        )
        dst_points = np.float32(
            [[8, 3], [4, rows - 17], [cols / 2 + 10, 15], [cols / 2 + 8, rows - 26]]
        )
        projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        X = np.array(
            [cv2.warpPerspective(x, projective_matrix, (cols, rows)) for x in X]
        )
    if args.affine is True:
        print("Applying affine transformation...")

        rows, cols = X.shape[1], X.shape[2]
        src_points = np.float32(
            [[rows // 4, cols // 4], [rows, cols // 4], [rows // 4, cols - 16]]
        )
        dst_points = np.float32(
            [[rows // 6, cols // 2], [rows, cols // 4], [rows // 2, cols]]
        )
        affine_matrix = cv2.getAffineTransform(src_points, dst_points)

        X = np.array([cv2.warpAffine(x, affine_matrix, (cols, rows)) for x in X])

    if args.deltah != 0 or args.deltas != 0 or args.deltav != 0:
        print("Applying color transformation in HSV space...")
        X = np.array([hsv_change(x, args.deltah, args.deltas, args.deltav) for x in X])

    np.savez_compressed(
        join(sprites_dir, filename), X, Y, S, beta_params, angle_params, shape_probs
    )
