import torch
import os
import numpy as np
from matplotlib import cm, pyplot as plt
from torchvision.utils import save_image
from metrics import mcc as met
import imageio
from torch.autograd import functional

def cart2pol(x, y):
    '''
    From cartesian to polar coordinates
    '''
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def batch_jacobian(func, x, create_graph=False):
  # x in shape (Batch, Length)
  def _func_sum(x):
    return func(x).sum(dim=0)
  return functional.jacobian(_func_sum, x, create_graph=create_graph).permute(1,0,2)

def scatterplot_variables(x, title, colors='None', cmap='hsv'):
    if colors=='None':
        plt.scatter(x[:,0], x[:,1], color='r', s=30)
    else:
        plt.scatter(x[:,0], x[:,1], c=colors, s=30, alpha=0.75, cmap=cmap)

    plt.axis('off')
    plt.gca().set_aspect('equal', adjustable='box')

def build_moebius_transform(alpha, A, a, b, epsilon=2):
    '''
    Implements Möbius transformations for D>=2, based on:
    https://en.wikipedia.org/wiki/Liouville%27s_theorem_(conformal_mappings)
    
    alpha: a scalar
    A: an orthogonal matrix
    a, b: vectors in R^D (dimension of the data)
    '''
    from jax import numpy as jnp
    def mixing_moebius_transform(x):
        if epsilon==2:
            frac = jnp.sum((x-a)**2)
            frac = frac**(-1)
        else:
            diff = jnp.abs(x-a)
            
            frac = 1.0
        return b + frac * alpha * A @ (x - a)
    
    B = jnp.linalg.inv(A)
    
    def unmixing_moebius_transform(y):
        numer = 1/alpha * (y - b)
        if epsilon==2:
            denom = jnp.sum((numer)**2)
        else:
            denom = 1.0
        return a + 1.0/denom * B @ numer
    
    return mixing_moebius_transform, unmixing_moebius_transform

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

def get_interp_name(args):
  return "latent_interpolations_" + "shape_" + str(int(args.shape)) + "_angle_" + str(int(args.angle)) + "_diag_" + str(args.diag) + "_seed_" + str(args.seed) + "_beta_" + str(args.beta)

def get_save_name(args):
  return "model_checkpoint_" + "dset_" + args.dset + "_shape_" + str(int(args.shape)) + "_angle_" + str(int(args.angle)) + "_diag_" + str(args.diag) + "_seed_" + str(args.seed) + "_beta_" + str(args.beta) + '.pth'

def get_load_name(args,train=True):
  if train:
    data = "train_"
  else:
    data = "val_"
  return "isprites_" + data + "nclasses_" + str(args.n_classes) + "_nobs_" + str(int(args.n_obs)) + "_lower_2_upper_15" + '.npz'

def get_corr_mat(net,data_loader,corr_type,epoch=None):
  true_factors = []
  estimated_factors = []
  net.eval()
  with torch.no_grad():
    for i, (x, l, f) in enumerate(data_loader):
        _, z, _ = net(x.to(net.posterior.device))
        true_factors.append(f.numpy())
        estimated_factors.append(z.cpu().numpy())

    true = torch.from_numpy(np.concatenate(true_factors)).permute(1,0).numpy()
    estimated = torch.from_numpy(np.concatenate(estimated_factors)).permute(1,0).numpy()
    if (epoch % 100) == 0:
      true_plot = torch.from_numpy(true).permute(1,0).numpy()
      estimated_plot = torch.from_numpy(estimated).permute(1,0).numpy()
      _, colors = cart2pol(true_plot[:, 0], true_plot[:, 1])
      estimated_plot[:,0] = estimated_plot[:,0]*-1
      scatterplot_variables(estimated_plot, 'Sources (estimated)', colors=colors)
      plt.title('Estimated (Epoch '+ str(epoch) + ")", fontsize=19)
      plt.savefig("Estimated_sources_mobius_epoch_" + str(epoch),dpi=150,bbox_inches='tight')
      plt.close()

    
    mat, _, _ = met.correlation(true, estimated, method=corr_type)
  return mat

def get_latent_interp(net):
  with torch.no_grad():
    net.eval()
    x = net.interp_sample
    decoder = net.decoder
    params = net.encoder(x.unsqueeze(0)).squeeze()
    mu = params[:net.z_dim]
    if net.posterior.diag:
      std = params[net.z_dim:].exp().sqrt()
    else:
      cholesky = torch.zeros((net.z_dim,net.z_dim)).to(x.device)
      cholesky_factors = params[net.z_dim:]
      it = 0
      for i in range(cholesky.shape[1]):
        for j in range(i+1):
          cholesky[i,j] = cholesky_factors[it]
          it+=1
      cov = torch.matmul(cholesky,cholesky.t())
      std = cov.diag().sqrt()

    gifs = []
    r = np.arange(-32,33,8)
    samples = []
    for row in range(net.z_dim):
      mean = mu[row].clone()
      sig = std[row].clone()
      z = mu.clone()
      for i in range(len(r)):
        z[row] = mean + sig*r[i]
        sample = decoder(torch.sigmoid(z))
        samples.append(sample)
        gifs.append(sample)
    samples = torch.cat(samples, dim=0).cpu()
    net.interp_dir = os.path.join(net.interp_dir, str(net.iter))
    os.makedirs(net.interp_dir, exist_ok=True)
    gifs = torch.cat(gifs)
    gifs = gifs.view(1, net.z_dim, len(r), x.shape[0], x.shape[1], x.shape[2]).transpose(1, 2)
    for j in range(len(r)):
        save_image(tensor=gifs[0][j].cpu(),
                  fp=os.path.join(net.interp_dir, '{}.jpg'.format(j)),
                  nrow=net.z_dim, pad_value=1)
    images = []
    for j in range(len(r)):
      filename = os.path.join(net.interp_dir, '{}.jpg'.format(j))
      images.append(imageio.imread(filename))
    
    out = os.path.join(net.interp_dir, '{}.gif'.format(j))
    imageio.mimsave(out, images)


