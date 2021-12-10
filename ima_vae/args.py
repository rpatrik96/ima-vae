import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='')


    # W and B
    parser.add_argument('--use-wandb', action='store_true', help="Log with Weights&Biases")
    parser.add_argument("--project", type=str, default="experiment",
                        help="This is the name of the experiment on Weights and Biases")
    parser.add_argument("--notes", type=str, default=None, help="Notes for the run on Weights and Biases")
    parser.add_argument("--tags", type=str,
                        nargs="*",  # 0 or more values expected => creates a list
                        default=None, help="Tags for the run on Weights and Biases")

    return parser