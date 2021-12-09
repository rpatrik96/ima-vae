import torch
import numpy as np
import torch

from ima_vae.data.data_generators import gen_data
from ima_vae.models.ivae.ivae_wrapper import IVAE_wrapper


def run_ivae_exp(args):
    """run iVAE simulations"""

    n_obs_per_seg = int(args.n_obs / args.n_segments)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.linear:
        activation = 'none'
    else:
        activation = 'lrelu'

    x, y, s = gen_data(Ncomp=args.latent_dim, Nlayer=args.n_layers, Nsegment=args.n_segments,
                       NsegmentObs=n_obs_per_seg,
                       orthog=args.orthog,
                       mobius=args.mobius,
                       seed=args.seed,
                       NonLin=activation)

    # Train/validation split
    p = np.random.permutation(len(x))
    x = x[p]
    y = y[p]
    s = s[p]

    X_tr, X_v = np.split(x, [int(.8 * len(x))])
    Y_tr, Y_v = np.split(y, [int(.8 * len(y))])
    S_tr, S_v = np.split(s, [int(.8 * len(s))])

    IVAE_wrapper(X=X_tr, U=Y_tr, S=S_tr, X_val=X_v, U_val=Y_v, S_val=S_v, n_layers=args.n_layers,
                 lr=args.lr,
                 max_iter=args.n_iter,
                 seed=args.seed,
                 batch_size=args.batch_size,
                 hidden_dim=args.latent_dim * 2,
                 activation=activation)
