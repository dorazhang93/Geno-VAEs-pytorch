from models import BaseVAE
from .types_ import *
from models.module import Encoders
from models.module import Decoders
from .loss import reconstruction_loss, KLD_loss, aux_loss
import numpy as np
import torch
from torch import nn

AUX_DIM=3
class VAE(BaseVAE):
    """
    Vanilla VAE and beta VAE
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
                 aux: bool = False,
                 **kwargs):
        super(VAE, self).__init__()
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
        if self.aux:
            self.aux_net = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
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
        if self.aux:
            aux_label = input[1]
            input = input[0]

        input=input.type(torch.float32)
        # for imputation evaluation
        # latent_dist,mask = self.encode(input)
        latent_dist = self.encode(input)
        # add noise
        z = self.reparameterize(*latent_dist)
        recons = self.decode(z)
        if self.aux:
            aux_logit = self.aux_net(z)
            return [recons , input, z, latent_dist, aux_label, aux_logit]
        return [recons , input, z, latent_dist]
        # for imputation evaluation
        # return [recons , input, z, latent_dist,mask]

    def loss_function(self, *args, **kwargs) -> Tensor:
        recons = args[0]
        y_true =args[1]
        latent_dist= args[3]
        recons_loss = reconstruction_loss(recons,y_true,self.loss)
        reg_loss = KLD_loss(latent_dist)
        kld_weight = self.latent_dim / self.in_dim
        loss = recons_loss + self.beta * kld_weight * reg_loss  #https://openreview.net/forum?id=Sy2fzU9gl
        if self.aux:
            aux_label = args[4]
            aux_logit = args[5]
            a_loss = aux_loss(aux_logit,aux_label)
            loss += a_loss * 0.001
        return {'loss': loss, 'Reconstruction_loss': recons_loss.detach(), 'Reg_loss':reg_loss.detach()}