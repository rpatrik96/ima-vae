import argparse
import os
import pickle
import torch
from runners.simulation_runner import run_ivae_exp

def parse_sim():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--orthog', type=bool, default=False, help='Whether mixing columns should be orthogonal')
    parser.add_argument('--mobius', type=bool, default=False, help='Whether mixing should be a mobius transform')
    parser.add_argument('--linear', type=bool, default=False, help='Whether mixing should be linear')
    parser.add_argument('--latent_dim', type=int, default=2, help='Latent/data dimension')
    parser.add_argument('--n_segments', type=int, default=40, help='Number of clusters in latent space')
    parser.add_argument('--n_layers', type=int, default=1, help='Number of layers in mixing')
    parser.add_argument('--n_obs', type=int, default=60e3, help='Number of observations in dataset')
    parser.add_argument('--n_iter', type=int, default=10e7, help='Maximum number of training iterations')
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')
    parser.add_argument("--seed", type=int, default=1, help="Seed for Random Number Generators")
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_sim()
    run_ivae_exp(args)



