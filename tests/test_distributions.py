import torch

from ima_vae.distributions import Beta


def test_beta_uniform():
    pdf = Beta()
    alpha = 1
    beta = 1
    z = torch.rand(128, 1)

    log_pdf = pdf.log_pdf(z, alpha * torch.ones((1, 1)), beta * torch.ones((1, 1)))

    assert not torch.any(torch.isnan(log_pdf)) and torch.all(log_pdf == 0)
