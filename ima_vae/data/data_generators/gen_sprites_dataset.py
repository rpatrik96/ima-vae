import argparse

import numpy as np
import torch

from spriteworld import environment as spriteworld_environment
from spriteworld import factor_distributions as distribs
from spriteworld import renderers as spriteworld_renderers
from spriteworld import sprite_generators
from spriteworld import tasks

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--nclasses", type=int, default=1, help="Number of auxiliary variables")
parser.add_argument("--nobs", type=int, default=10000, help="Number of observations in dataset")
parser.add_argument("--lower", type=int, default=2, help="Lower bound on alpha and beta (Set to at least 2)")
parser.add_argument("--upper", type=int, default=15, help="Upper bound on alpha and beta")
parser.add_argument("--angle", type=bool, default=False, help="True if you want angle as a factor")
parser.add_argument("--shape", type=bool, default=False, help="True if you want shape as a factor")
args = parser.parse_args()


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


nfactors = 4
beta_params = torch.Tensor(np.random.uniform(args.lower, args.upper, 2 * nfactors * args.nclasses)).view(args.nclasses,
                                                                                                         nfactors,
                                                                                                         2).numpy()
angle_params = torch.zeros((args.nclasses, 2)).numpy()
shape_probs = torch.zeros((args.nclasses, 3)).numpy()


sname = "isprites_" + "nclasses_" + str(args.nclasses) + "_nobs_" + str(args.nobs) + "_lower_" + str(
    args.lower) + "_upper_" + str(args.upper)
sname_train = "isprites_train_" + "nclasses_" + str(args.nclasses) + "_nobs_" + str(args.nobs) + "_lower_" + str(
    args.lower) + "_upper_" + str(args.upper)
sname_val = "isprites_val_" + "nclasses_" + str(args.nclasses) + "_nobs_" + str(args.nobs) + "_lower_" + str(
    args.lower) + "_upper_" + str(args.upper)
if args.angle:
    nfactors += 1
    sname += "_angle"
if args.shape:
    nfactors += 1
    sname += "_shape"


def random_sprites_config(beta_params, label):
    factor_list = [
        distribs.Beta('x', beta_params[label][0][0], beta_params[label][0][1]),
        distribs.Beta('y', beta_params[label][1][0], beta_params[label][1][1]),
        distribs.Beta('scale', beta_params[label][2][0], beta_params[label][2][1]),
        # We are using HSV, so "c0 = H", "c1 = S", "c2 = V"
        distribs.Beta('c0', beta_params[label][3][0], beta_params[label][3][1]),
        distribs.Continuous('c1', 1., 1.),
        distribs.Continuous('c2', 1., 1.)]

    if args.angle:
        angles = np.random.uniform(args.lower, args.upper, 2)
        angle_params[label] = angles
        factor_list.append(distribs.Beta('angle', angle_params[label][0], angle_params[label][1]))

    if args.shape:
        probs = np.random.uniform(0, 1, 3)
        probs = probs / probs.sum()
        shape_probs[label] = probs
        factor_list.append(distribs.Discrete('shape', ['triangle', 'square', 'pentagon'], probs=shape_probs[label]))
    else:
        factor_list.append(distribs.Discrete('shape', ['triangle']))

    factors = distribs.Product(factor_list)
    sprite_gen = sprite_generators.generate_sprites(factors, num_sprites=1)

    renderers = {
        'image':
            spriteworld_renderers.PILRenderer(
                image_size=(64, 64),
                anti_aliasing=5,
                color_to_rgb=spriteworld_renderers.color_maps.hsv_to_rgb,
            ),
        'attributes':
            spriteworld_renderers.SpriteFactors(
                factors=('x', 'y', 'shape', 'angle', 'scale', 'c0', 'c1', 'c2')),
    }

    config = {
        'task': tasks.NoReward(),
        'action_space': None,
        'renderers': renderers,
        'init_sprites': sprite_gen,
        'max_episode_length': 1,
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
            if env._sprites[0].shape == 'triangle':
                S[label, i, 5] = 0
            elif env._sprites[0].shape == 'square':
                S[label, i, 5] = 1
            elif env._sprites[0].shape == 'pentagon':
                S[label, i, 5] = 2

        images.append(ts.observation['image'])
    return images


def generate_isprites(num_classes, obs_per_class, lower, upper):
    for i in range(num_classes):
        print(i)
        if i == 0:
            full_obs = collect_frames(random_sprites_config(beta_params, i), i, obs_per_class)
            full_labels = np.zeros(obs_per_class)
        else:
            full_obs += collect_frames(random_sprites_config(beta_params, i), i, obs_per_class)
            full_labels = np.concatenate((full_labels, np.ones(obs_per_class) * i))

    return np.array(full_obs), np.array(full_labels)


obs_per_class = int(args.nobs / args.nclasses)
S = np.zeros((args.nclasses, obs_per_class, nfactors))
X, Y = generate_isprites(args.nclasses, obs_per_class, args.lower, args.upper)
S = torch.Tensor(S).flatten(0, 1).numpy().astype(np.float32)
Y = to_one_hot(Y)[0].astype(np.float32)

# Train/validation split
# p = np.random.permutation(len(X))
# X = X[p]
# Y = Y[p]
# S = S[p]
#
# X_tr, X_v = np.split(X, [int(.8 * len(X))])
# Y_tr, Y_v = np.split(Y, [int(.8 * len(Y))])
# S_tr, S_v = np.split(S, [int(.8 * len(S))])

np.savez_compressed(sname, X, Y, S, beta_params, angle_params, shape_probs)
# np.savez_compressed(sname_train, X_tr, Y_tr, S_tr, beta_params, angle_params, shape_probs)
# np.savez_compressed(sname_val, X_v, Y_v, S_v, beta_params, angle_params, shape_probs)
