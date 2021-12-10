import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--n_iter', type=int, default=10e7, help='Maximum number of training iterations')
    parser.add_argument("--seed", type=int, default=1, help="Seed for Random Number Generators")

    return parser.parse_args()