from ima_vae.args import parse_sim
from runners.simulation_runner import run_ivae_exp

if __name__ == '__main__':
    args = parse_sim()
    run_ivae_exp(args)
