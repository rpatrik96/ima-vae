import numpy as np
import torch
import torchvision.transforms as T
from torch import nn
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import save_image
import distributions as dist
from models import vaes
import utils
from data import dataloaders
from data.data_generators.gen_synth_dataset import gen_data
from utils import to_one_hot, cart2pol, scatterplot_variables, get_load_name

def train_ivae(args):

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  sname = utils.get_save_name(args)

  # linear or nonlinear generative model
  if args.linear:
      activation = 'none'
  else:
      activation = 'lrelu'

  # if sprites dataset, getting filename to save latent interpolations
  if args.dset == 'image':
    interpname = utils.get_interp_name(args)

  # specifying prior, posterior, and likelihood
  if args.prior == 'beta':
    prior = dist.Beta()
  elif args.prior == 'gaussian':
    prior = dist.Normal(device=device)
  elif args.prior == 'uniform':
    prior = dist.Uniform()
  posterior = dist.Normal(device=device,diag=bool(args.diag))
  likelihood = dist.Normal(device=device)

  # initialize iVAE
  net = vaes.iVAE(prior=prior,
    posterior=posterior,
    likelihood=likelihood,
    z_dim=args.z_dim,
    nclasses=args.n_classes,
    dset=args.dset,
    activation=activation,
    device=device,
    n_layers=args.n_layers).to(device)

  # create optimizer
  optimizer = optim.Adam(net.parameters(),lr=args.lr)

  # load model and optimizer state if specified
  if args.lname != None:
      net.load_state_dict(torch.load(args.lname)['model'])
      optimizer.load_state_dict(torch.load(args.lname)['optimizer'])

  '''
  Load dataset: If sprites dataset we are loading pre-generated data from a file. 
  If synthetic dataset we are creating dataset.
  '''
  if args.dset == 'image':
    train_loader = dataloaders.get_dataloader(args.n_batch,lname=get_load_name(args,train=True),transform=True)
    val_loader = dataloaders.get_dataloader(args.n_batch,lname=get_load_name(args,train=False),transform=True)
  elif args.dset == 'synth':
      n_obs_per_seg = int(args.n_obs/args.n_classes)
      x, y, s = gen_data(Ncomp=args.z_dim, Nlayer=args.n_layers, Nsegment=args.n_classes, 
                  NsegmentObs=n_obs_per_seg, 
                  orthog=args.orthog,
                  mobius=args.mobius, 
                  seed=args.seed, 
                  NonLin=activation,
                  source=args.prior)

      # Train/validation split
      p = np.random.permutation(len(x))
      x = x[p]
      y = y[p]
      s = s[p]
      
      X_tr,X_v = np.split(x, [int(.8 * len(x))])
      Y_tr,Y_v = np.split(y, [int(.8 * len(y))])
      S_tr,S_v = np.split(s, [int(.8 * len(s))])

      train_loader = dataloaders.get_dataloader(batch_size=args.n_batch,
        X=X_tr.astype(np.float32),
        Y=Y_tr.astype(np.float32),
        S=S_tr.astype(np.float32),
        transform=False)

      val_loader = dataloaders.get_dataloader(batch_size=args.n_batch,
        X=X_v.astype(np.float32),
        Y=Y_v.astype(np.float32),
        S=S_v.astype(np.float32),
        transform=False)

  # Training loop for model
  net.train()
  max_epochs = int(args.n_iter // len(train_loader) + 1)
  mccs = []
  while net.iter < args.n_iter:
    elbo_train = 0
    epoch = net.iter // len(train_loader) + 1
    for _, (x, u, s) in enumerate(train_loader):
      net.iter += 1
      optimizer.zero_grad()
      x_recon, z, log_qz_xu = net(x.to(device))

      # observation likelihood
      log_px_z = net.likelihood.log_pdf(x_recon.flatten(1), x.to(device).flatten(1), .00001 * torch.ones(1).to(device))
      #log_px_z = ((x_recon-x.to(device))**2).flatten(1).sum(1).mul(-1)

      # all prior parameters fixed if uniform
      if prior.name != 'uniform':
        prior_params = net.conditioner(u.to(device))

      # prior likelihood
      if prior.name == 'gauss':
        log_pz_u = net.prior.log_pdf(z, prior_params[:,:args.z_dim], prior_params[:,args.z_dim:].exp())
        #log_pz_u = net.prior.log_pdf(z, torch.zeros(1).to(device), prior_params.exp())
      elif prior.name == 'beta':
        log_pz_u = net.prior.log_pdf(z, prior_params[:,:args.z_dim], prior_params[:,args.z_dim:])
        #log_pz_u = net.prior.log_pdf(z, torch.ones((z.shape[0],args.z_dim))*3, torch.ones((z.shape[0],args.z_dim))*11)
      elif prior.name == 'uniform':
        log_pz_u = net.prior.log_pdf(z, torch.zeros(1).to(device), torch.ones(1).to(device))

      # calculate mean negative elbo for batch 
      neg_elbo = (log_px_z + args.beta*(log_pz_u - log_qz_xu)).mean().mul(-1)
      neg_elbo.backward()
      optimizer.step()
      elbo_train += neg_elbo.item()

      # Get sample that will be used for latent interpolations throughout training
      if net.iter == 1 and args.dset == 'image':
        net.interp_sample = x[0].to(device)
        net.interp_dir = interpname
    
      # Print Stats
      if net.iter % (len(train_loader)) == 0:
        elbo_train /= len(train_loader)
        mat = utils.get_corr_mat(net,val_loader,corr_type=args.corr,epoch=epoch)
        ccs = np.abs(np.diag(mat))
        mcc = np.mean(ccs)
        mccs.append(mcc)
        np.save("mccs_z_dim_" + str(args.z_dim), mccs)
        print("Epoch: ", epoch, "Loss: ", elbo_train, "MCC: ", mcc)
        if (args.dset == 'image') and net.iter!=0:
          utils.get_latent_interp(net)
        net.train()

      # Save model
      if net.iter % args.savestep == 0 and net.iter!=0:
        checkpoint = { 'model': net.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, sname)
      
