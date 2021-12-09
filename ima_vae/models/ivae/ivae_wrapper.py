import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

import ima_vae.metrics
from ima_vae.data.data_generators import ConditionalDataset
from .ivae_core import iVAE


def IVAE_wrapper(X, U, S, X_val, U_val, S_val, n_layers, lr, max_iter, seed, batch_size, hidden_dim, activation,
                 ckpt_file='ivae_uncond.pt'):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    dset = ConditionalDataset(X.astype(np.float32), U.astype(np.float32), S.astype(np.float32), device)
    dset_val = ConditionalDataset(X_val.astype(np.float32), U_val.astype(np.float32), S_val.astype(np.float32), device)
    train_loader = DataLoader(dset, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(dset_val, shuffle=True, batch_size=batch_size)

    data_dim, latent_dim, aux_dim = dset.get_dims()
    N = len(dset)
    max_epochs = int(max_iter // len(train_loader) + 1)

    # define model and optimizer
    model = iVAE(latent_dim=latent_dim,
                 data_dim=data_dim,
                 aux_dim=aux_dim,
                 n_layers=n_layers,
                 hidden_dim=hidden_dim,
                 activation=activation,
                 device=device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # training loop
    print("Training..")
    it = 0
    # mccs = []
    # model.load_state_dict(torch.load('ivae_uncond.pt'))
    model.train()
    while it < max_iter:
        elbo_train = 0
        epoch = it // len(train_loader) + 1
        for _, (obs, labels, sources) in enumerate(train_loader):
            it += 1
            optimizer.zero_grad()
            obs, labels = obs.to(device), labels.to(device)
            elbo, z_est = model.elbo(obs, labels)
            elbo.mul(-1).backward()
            optimizer.step()
            elbo_train += -elbo.item()
        elbo_train /= len(train_loader)
        print('epoch {}/{} \tloss: {}'.format(epoch, max_epochs, elbo_train))

        # Get MCC for this epoch on validation set
        if ((epoch % 1) == 0) or (it == max_iter):
            true_factors = []
            estimated_factors = []
            model.eval()
            with torch.no_grad():
                for _, (obs, labels, sources) in enumerate(val_loader):
                    obs, labels, sources = obs.to(device), labels.to(device), sources.to(device)
                    _, z = model.elbo(obs, labels)
                    true_factors.append(sources.numpy())
                    estimated_factors.append(z.cpu().numpy())
                true = torch.from_numpy(np.concatenate(true_factors)).permute(1, 0).numpy()
                estimated = torch.from_numpy(np.concatenate(estimated_factors)).permute(1, 0).numpy()
                mat, _, _ = ima_vae.metrics.mcc.correlation(true, estimated, method='Pearson')
                mcc = np.mean(np.abs(np.diag(mat)))
                print('MCC:', mcc)
                # mccs.append(mcc)
                # np.save('ivae_mccs_uncond',np.array(mccs))
                model.train()

        # torch.save(model.state_dict(), ckpt_file)
