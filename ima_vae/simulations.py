from ima_vae.args import parse_args
from ima_vae.runners.simulation_runner import run_ivae_exp

if __name__ == '__main__':
    args = parse_args()
    run_ivae_exp(args)
