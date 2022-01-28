import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
import distributions as dist
from models import nets

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)

class iVAE(nn.Module):
    def __init__(self, prior, posterior, likelihood, z_dim, nclasses, dset, activation, device, n_layers, slope=.2, nc=3):
        super(iVAE, self).__init__()
        self.z_dim = z_dim
        self.x_dim = z_dim
        self.prior = prior
        self.posterior = posterior
        self.likelihood = likelihood
        if posterior.diag:
          self.post_dim = 2*z_dim
        else:
          self.cholesky_factors = (z_dim*(z_dim+1))/2
          self.post_dim = int(z_dim + self.cholesky_factors)
          self.cholesky = None

        if dset == 'synth':
          self.encoder = nets.MLP(self.x_dim, self.post_dim, z_dim*10, n_layers, activation=activation, slope=slope, device=device)
          self.decoder = nets.MLP(z_dim, self.x_dim, z_dim*10, n_layers, activation=activation, slope=slope, device=device)

        elif dset == 'image':
          self.encoder, self.decoder = nets.get_sprites_models(self.z_dim,self.post_dim,nc=3)

        if self.prior.name != 'uniform':
            self.conditioner = nets.MLP(nclasses, z_dim*2, z_dim*4, n_layers, activation=activation, slope=slope, device=device)

        self.interp_sample = None
        self.interp_dir = None
        self.iter = 0
        self.apply(weights_init)

    def populate_cholesky(self,cholesky_factors):
      it = 0
      for i in range(self.cholesky.shape[1]):
        for j in range(i+1):
          self.cholesky[:,i,j] = cholesky_factors[:,it]
          it+=1

    def forward(self, x):
        params = self.encoder(x)
        mu = params[:, :self.z_dim]
        
        if self.posterior.diag:
          logvar = params[:, self.z_dim:]
          z = self.posterior.sample(mu, logvar.exp())
          log_qz_u = self.posterior.log_pdf(z,mu,logvar.exp())
        else:
          self.cholesky = torch.zeros((x.shape[0],self.z_dim,self.z_dim)).to(x.device)
          cholesky_factors = params[:, self.z_dim:]
          self.populate_cholesky(cholesky_factors)
          z = self.posterior.sample(mu, self.cholesky)
          log_qz_u = self.posterior.log_pdf_full(z, mu, self.cholesky)

        if self.prior.name == 'beta' or self.prior.name == 'uniform':
          determ = torch.log(1 / (torch.sigmoid(z) * (1 - torch.sigmoid(z)))).sum(1)
          log_qz_u += determ
          z = torch.sigmoid(z)

        x_recon = self.decoder(z)
        return x_recon, z, log_qz_u

