from models import BaseVAE
from .types_ import *
from models.module import Encoders
from models.module import Decoders
from .loss import reconstruction_loss,aux_loss
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

class idVAE(BaseVAE):
    """
    identifiable double VAE: VAE and nonlinear ICA, hierarchical VAE
    http://proceedings.mlr.press/v139/mita21a/mita21a.pdf
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
        super(idVAE, self).__init__()
        self.sparsify = sparsify
        self.beta = beta
        self.missing_val = missing_val
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.in_dim = in_dim
        self.loss = loss
        self.aux = aux

        print(f"%%% building idVAE with {encoder} and {decoder}%%%")
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
        self.u_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Dropout(0.01),
            nn.Linear(hidden_dim,AUX_DIM)
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

        # second layer of VAE
        u_sample = self.reparameterize(*prior_dist)
        recon_u = self.u_decoder(u_sample)
        return [recons , input, z_sample, recon_u, aux_label, post_dist,prior_dist]
        # for evaluating p(z|x)
        # return [recons , input, latent_dist[0], recon_u, aux_label, post_dist,prior_dist]

    def loss_function(self, *args, **kwargs) -> Tensor:
        recons = args[0]
        y_true =args[1]
        recon_u =args[3]
        aux_label = args[4]
        post_dist= args[5]
        prior_dist = args[6]
        recons_loss = reconstruction_loss(recons,y_true,self.loss)
        kl_post_prior = kl_criterion(*post_dist, *prior_dist)
        kld_weight = self.latent_dim / self.in_dim
        loss = recons_loss + self.beta * kld_weight * kl_post_prior
        loss += aux_loss(recon_u, aux_label)*0.1
        loss += kl_criterion(*prior_dist, torch.zeros_like(prior_dist[0]), torch.ones_like(prior_dist[1]))*0.1
        return {'loss': loss, 'Reconstruction_loss': recons_loss.detach(), 'Reg_loss':kl_post_prior.detach()}