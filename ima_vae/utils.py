import os

import imageio
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn as nn
from torch.autograd import functional
from torchvision.utils import save_image

from ima_vae.data.utils import cart2pol, scatterplot_variables
from ima_vae.metrics import mcc


def batch_jacobian(func, x, create_graph=False):
    # x in shape (Batch, Length)
    def _func_sum(x):
        return func(x).sum(dim=0)

    return functional.jacobian(_func_sum, x, create_graph=create_graph).permute(1, 0, 2)


def get_interp_name(args):
    return "latent_interpolations_" + "shape_" + str(int(args.shape)) + "_angle_" + str(
        int(args.angle)) + "_diag_" + str(args.diag) + "_seed_" + str(args.seed) + "_beta_" + str(args.beta)


def get_save_name(args):
    return "model_checkpoint_" + "dset_" + args.dset + "_shape_" + str(int(args.shape)) + "_angle_" + str(
        int(args.angle)) + "_diag_" + str(args.diag) + "_seed_" + str(args.seed) + "_beta_" + str(args.beta) + '.pth'


def get_corr_mat(net, data_loader, corr_type, epoch=None):
    true_factors = []
    estimated_factors = []
    net.eval()
    with torch.no_grad():
        for i, (x, l, f) in enumerate(data_loader):
            _, z, _ = net(x.to(net.posterior.device))
            true_factors.append(f.numpy())
            estimated_factors.append(z.cpu().numpy())

        true = torch.from_numpy(np.concatenate(true_factors)).permute(1, 0).numpy()
        estimated = torch.from_numpy(np.concatenate(estimated_factors)).permute(1, 0).numpy()
        if (epoch % 100) == 0:
            true_plot = torch.from_numpy(true).permute(1, 0).numpy()
            estimated_plot = torch.from_numpy(estimated).permute(1, 0).numpy()
            _, colors = cart2pol(true_plot[:, 0], true_plot[:, 1])
            estimated_plot[:, 0] = estimated_plot[:, 0] * -1
            scatterplot_variables(estimated_plot, 'Sources (estimated)', colors=colors)
            plt.title('Estimated (Epoch ' + str(epoch) + ")", fontsize=19)
            plt.savefig("Estimated_sources_mobius_epoch_" + str(epoch), dpi=150, bbox_inches='tight')
            plt.close()

        mat, _, _ = mcc.correlation(true, estimated, method=corr_type)
    return mat


def get_latent_interp(net):
    with torch.no_grad():
        net.eval()
        x = net.interp_sample
        decoder = net.decoder
        params = net.encoder(x.unsqueeze(0)).squeeze()
        mu = params[:net.latent_dim]
        if net.posterior.diag:
            std = params[net.latent_dim:].exp().sqrt()
        else:
            cholesky = torch.zeros((net.latent_dim, net.latent_dim)).to(x.device)
            cholesky_factors = params[net.latent_dim:]
            it = 0
            for i in range(cholesky.shape[1]):
                for j in range(i + 1):
                    cholesky[i, j] = cholesky_factors[it]
                    it += 1
            cov = torch.matmul(cholesky, cholesky.t())
            std = cov.diag().sqrt()

        gifs = []
        r = np.arange(-32, 33, 8)
        samples = []
        for row in range(net.latent_dim):
            mean = mu[row].clone()
            sig = std[row].clone()
            z = mu.clone()
            for i in range(len(r)):
                z[row] = mean + sig * r[i]
                sample = decoder(torch.sigmoid(z))
                samples.append(sample)
                gifs.append(sample)
        samples = torch.cat(samples, dim=0).cpu()
        net.interp_dir = os.path.join(net.interp_dir, str(net.iter))
        os.makedirs(net.interp_dir, exist_ok=True)
        gifs = torch.cat(gifs)
        gifs = gifs.view(1, net.latent_dim, len(r), x.shape[0], x.shape[1], x.shape[2]).transpose(1, 2)
        for j in range(len(r)):
            save_image(tensor=gifs[0][j].cpu(),
                       fp=os.path.join(net.interp_dir, '{}.jpg'.format(j)),
                       nrow=net.latent_dim, pad_value=1)
        images = []
        for j in range(len(r)):
            filename = os.path.join(net.interp_dir, '{}.jpg'.format(j))
            images.append(imageio.imread(filename))

        out = os.path.join(net.interp_dir, '{}.gif'.format(j))
        imageio.mimsave(out, images)


def calc_jacobian(model: nn.Module, latents: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Jacobian more efficiently than ` torch.autograd.functional.jacobian`
    :param model: the model to calculate the Jacobian of
    :param latents: the inputs for evaluating the model
    :return: B x n_out x n_in
    """

    jacob = []
    input_vars = latents.clone().requires_grad_(True)

    # set to eval mode but remember original state
    in_training: bool = model.training
    model.eval()  # otherwise we will get 0 gradients
    with torch.set_grad_enabled(True):

        output_vars = model(input_vars).flatten(1)

        for i in range(output_vars.shape[1]):
            jacob.append(torch.autograd.grad(output_vars[:, i:i + 1], input_vars, create_graph=True,
                                             grad_outputs=torch.ones(output_vars[:, i:i + 1].shape).to(
                                                 output_vars.device))[
                             0])

        jacobian = torch.stack(jacob, 1)

    # set back to original mode
    if in_training is True:
        model.train()

    return jacobian.mean(0)
