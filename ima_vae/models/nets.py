from typing import Literal

from torch import nn
from torch.nn import functional as F

ActivationType = Literal["lrelu", "sigmoid", "none"]


class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        n_layers,
        activation: ActivationType,
        device,
        slope=0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.device = device
        self.hidden_dim = [hidden_dim] * (self.n_layers - 1)
        self.activation = [activation] * (self.n_layers - 1)

        self._act_f = []
        for act in self.activation:
            if act == "lrelu":
                self._act_f.append(lambda x: F.leaky_relu(x, negative_slope=slope))
            elif act == "sigmoid":
                self._act_f.append(F.sigmoid)
            elif act == "none":
                self._act_f.append(lambda x: x)
            else:
                ValueError("Incorrect activation: {}".format(act))

        if self.n_layers == 1:
            _fc_list = [nn.Linear(self.input_dim, self.output_dim)]
        else:
            _fc_list = [nn.Linear(self.input_dim, self.hidden_dim[0])]
            for i in range(1, self.n_layers - 1):
                _fc_list.append(nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i]))
            _fc_list.append(
                nn.Linear(self.hidden_dim[self.n_layers - 2], self.output_dim)
            )
        self.fc = nn.ModuleList(_fc_list)
        self.to(self.device)

    def forward(self, x):
        h = x
        for c in range(self.n_layers):
            if c == self.n_layers - 1:
                h = self.fc[c](h)
            else:
                h = self._act_f[c](self.fc[c](h))
        return h


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


def get_sprites_models(z_dim, post_dim, n_channels=3):
    encoder = nn.Sequential(
        nn.Conv2d(n_channels, 32, 4, 2, 1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 4, 2, 1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 4, 2, 1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 4, 2, 1),
        nn.ReLU(),
        View((-1, 32 * 4 * 4)),
        nn.Linear(32 * 4 * 4, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, post_dim),
    )

    decoder = nn.Sequential(
        nn.Linear(z_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 32 * 4 * 4),
        nn.ReLU(),
        View((-1, 32, 4, 4)),
        nn.ConvTranspose2d(32, 32, 4, 2, 1),
        nn.ReLU(),
        nn.ConvTranspose2d(32, 32, 4, 2, 1),
        nn.ReLU(),
        nn.ConvTranspose2d(32, 32, 4, 2, 1),
        nn.ReLU(),
        nn.ConvTranspose2d(32, n_channels, 4, 2, 1),
    )

    return encoder, decoder


def get_synth_models(
    data_dim,
    latent_dim,
    hidden_dim,
    post_dim,
    n_layers,
    activation,
    device,
    slope,
    encoder_extra_layers=0,
    encoder_extra_width=0,
):
    encoder = MLP(
        data_dim,
        post_dim,
        hidden_dim + encoder_extra_width,
        n_layers + encoder_extra_layers,
        activation=activation,
        slope=slope,
        device=device,
    )
    decoder = MLP(
        latent_dim,
        data_dim,
        hidden_dim,
        n_layers,
        activation=activation,
        slope=slope,
        device=device,
    )

    return encoder, decoder
