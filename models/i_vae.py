from models import BaseVAE
from .types_ import *
from models.module import Encoders
from models.module import Decoders
from .loss import reconstruction_loss, log_Normal_pdf
import numpy as np
import torch
from torch import nn

AUX_DIM=3

def compute_posterior(z_x_mu, z_x_logv, z_u_mu, z_u_logv):
    # q(z|x,u) = q(z|x)q(z|u) = N((mu1*var2+mu2*var1)/(var1+var2), var1*var2/(var1+var2))
    post_mu = (z_x_mu/(1+torch.exp(z_x_logv-z_u_logv))) + (z_u_mu/(1+torch.exp(z_u_logv-z_x_logv)))
    post_logv = z_x_logv + z_u_logv - torch.log(torch.exp(z_x_logv)+torch.exp(z_u_logv))
    return post_mu, post_logv

def kl_criterion(mu1, logv1, mu2, logv2):
    sigma1 = logv1.mul(0.5).exp()
    sigma2 = logv2.mul(0.5).exp()
    kld = torch.log(sigma2/sigma1) + (torch.exp(logv1)+(mu1-mu2)**2)/(2*torch.exp(logv2)) - 0.5
    return torch.mean(kld)

class iVAE(BaseVAE):
    """
    identifiable VAE: VAE and nonlinear ICA
    https://arxiv.org/pdf/1907.04809.pdf
    """

    def __init__(self,
                 in_channels: int,
                 in_dim: int,
                 latent_dim: int,
                 hidden_dim: int,
                 encoder: str = "EncoderConv1D",
                 decoder: str = "DecoderConv1D",
                 beta: float = 1.0,
                 missing_val: float = -1.0,
                 sparsify: float = 0.4,
                 loss: str = "BCE",
                 aux: bool = True,
                 **kwargs):
        super(iVAE, self).__init__()
        self.sparsify = sparsify
        self.beta = beta
        self.missing_val = missing_val
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.in_dim = in_dim
        self.loss = loss
        self.aux = aux

        print(f"%%% building VAE with {encoder} and {decoder}%%%")
        self.encoder = Encoders[encoder](in_channels,in_dim,hidden_dim=hidden_dim,**kwargs)
        self.fc_latent = nn.Linear(hidden_dim,latent_dim*2)
        self.decoder = Decoders[decoder](latent_dim=latent_dim,out_channel=in_channels,out_dim=in_dim,hidden_dim=hidden_dim,
                                         **kwargs)
        self.prior = nn.Sequential(
            nn.Linear(AUX_DIM, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Dropout(0.01),
            nn.Linear(hidden_dim,latent_dim*2)
        )

    def encode(self, input):
        device= input.device
        x=input
        if self.training:
            spar = self.sparsify * np.random.uniform()
            msk = torch.from_numpy((np.random.random_sample(x.shape) < spar).astype(int)).to(device)
            x = x*(1-msk)+msk*self.missing_val
        x = torch.unsqueeze(x, dim=1)
        x = self.encoder(x)
        mu_logvar = self.fc_latent(x)
        mu, logvar = mu_logvar.view(-1,self.latent_dim, 2).unbind(-1)
        return mu, logvar
        # for imputation evaluation
        # return (mu, logvar), msk

    def decode(self,z):
        return self.decoder(z)

    def reparameterize(self, mu, logvar):
        # for imputation evaluation
        # return mu
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps * std + mu
        else:
    #         add no noise for eval mode
            return mu

    def forward(self, input: Tensor) -> Tensor:
        aux_label = input[1]
        input = input[0]
        input=input.type(torch.float32)
        # for imputation evaluation
        # latent_dist,mask = self.encode(input)
        latent_dist = self.encode(input)
        prior_dist = (self.prior(aux_label.type(torch.float32))).view(-1,self.latent_dim,2).unbind(-1)
        post_dist = compute_posterior(*latent_dist, *prior_dist)
        # add noise
        z_sample = self.reparameterize(*post_dist)
        recons = self.decode(z_sample)
        return [recons , input, z_sample, post_dist,prior_dist]
        # for evaluating p(z|x)
        # return [recons , input, latent_dist[0], post_dist,prior_dist]

    def loss_function(self, *args, **kwargs) -> Tensor:
        recons = args[0]
        y_true =args[1]
        post_dist= args[3]
        prior_dist = args[4]
        recons_loss = reconstruction_loss(recons,y_true,self.loss)
        kl_post_prior = kl_criterion(*post_dist, *prior_dist)
        kld_weight = self.latent_dim / self.in_dim
        loss = recons_loss + self.beta * kld_weight * kl_post_prior
        return {'loss': loss, 'Reconstruction_loss': recons_loss.detach(), 'Reg_loss':kl_post_prior.detach()}