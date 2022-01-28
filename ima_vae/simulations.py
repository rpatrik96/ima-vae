import argparse
import os
import pickle
import torch
import numpy as np
import random
from train_ivae import train_ivae
from inspect_jacobian import train_regression
from train_classifier import train_classifier

def parse_sim():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--orthog', type=bool, default=False, help='Whether mixing columns should be orthogonal')
    parser.add_argument('--mobius', type=bool, default=False, help='Whether mixing should be a mobius transform')
    parser.add_argument('--linear', type=bool, default=False, help='Whether mixing should be linear')
    parser.add_argument('--z_dim', type=int, default=5, help='Latent/data dimension')
    parser.add_argument("--n_classes", type=int, default=1, help="Number of auxiliary variables")
    parser.add_argument('--n_layers', type=int, default=2, help='Number of layers in mixing')
    parser.add_argument('--n_obs', type=int, default=100e3, help='Number of observations in dataset')
    parser.add_argument('--n_iter', type=int, default=10e7, help='Maximum number of training iterations')
    parser.add_argument('--n_batch', type=int, default=64, help='Training batch size')
    parser.add_argument("--seed", type=int, default=1, help="Seed for Random Number Generators")
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument("--dset", type=str, default="synth", help="Dataset to use (synth or image)")
    parser.add_argument("--prior", type=str, default="gaussian", help="Prior distribution")
    parser.add_argument("--diag", type=int, default=1, help="Whether or not posterior has diagonal covariance. 1 for True, 0 for False")
    parser.add_argument("--savestep", type=int, default=50, help="Iteration to save model")
    parser.add_argument("--lname", type=str, default=None, help="Name of model if you want to load a model")
    parser.add_argument("--corr", type=str, default="Pearson", help="Correlation type for MCC")
    parser.add_argument("--beta", type=int, default=1, help="Beta hyperparameter on the dkl")
    parser.add_argument("--angle", type=bool, default=False, help="True if you want angle as a factor for image dataset")
    parser.add_argument("--shape", type=bool, default=False, help="True if you want shape as a factor for image dataset")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_sim()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    train_ivae(args)

