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
from torch.autograd import functional
from data.data_generators.gen_synth_dataset import gen_data
from models import nets
from utils import to_one_hot, cart2pol, scatterplot_variables, get_load_name, batch_jacobian

def train_regression(args):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  if args.linear:
      activation = 'none'
  else:
      activation = 'lrelu'
  if args.dset == 'image':
    _, net = nets.get_sprites_models(args.z_dim,args.z_dim)
    train_loader = dataloaders.get_dataloader(args.n_batch,lname=get_load_name(args,train=True),transform=True)
    val_loader = dataloaders.get_dataloader(args.n_batch,lname=get_load_name(args,train=False),transform=True)
  elif args.dset == 'synth':
    net = nets.MLP(args.z_dim, args.z_dim, args.z_dim*10, args.n_layers, activation=activation, slope=.2, device=device)
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

  # create optimizer
  optimizer = optim.Adam(net.parameters(),lr=args.lr)
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

  # Training loop for model
  max_epochs = int(args.n_iter // len(train_loader) + 1)
  it = 0
  net.train()
  while it < args.n_iter:
    epoch = it // len(train_loader) + 1
    for _, (x, u, s) in enumerate(train_loader):
      it += 1
      optimizer.zero_grad()
      x_recon = net(s.to(device))

      # mean reconstruction
      loss = ((x_recon-x.to(device))**2).flatten(1).sum(1).mean()
      loss.backward()
      optimizer.step()

    scheduler.step()
    loss_val=0
    jacob = torch.zeros((s.shape[1],s.shape[1]))
    for _, (x, u, s) in enumerate(val_loader):
      net.eval()
      jacobians = batch_jacobian(net,s)
      jacob += torch.matmul(jacobians, torch.transpose(jacobians,1,2)).sum(0)/s.shape[0]
      x_recon = net(s.to(device))
      # mean reconstruction
      loss = ((x_recon-x.to(device))**2).flatten(1).sum(1).mean()
      loss_val += loss.item()
    jacob/=len(val_loader)
    orthog = (jacob.abs().sum() - jacob.abs().diag().sum())/((s.shape[1]*s.shape[1])-s.shape[1])
    loss_val /= len(val_loader)
    print("Epoch: ", epoch, "Loss: ", loss_val, "Orthog: ", orthog.item())
    net.train()

      