from models import BaseVAE
from .types_ import *
from models.module import Encoders
from models.module import Decoders
from .loss import reconstruction_loss, regulization_loss, aux_loss
import numpy as np
import torch
from torch import nn

AUX_DIM=3

class AE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 in_dim: int,
                 latent_dim: int,
                 hidden_dim: int,
                 encoder: str = "EncoderConv1D",
                 decoder: str = "DecoderConv1D",
                 noise_std: float = 0.032,
                 reg_factor: float = 1.0e-8,
                 missing_val: float = -1.0,
                 sparsify: float = 0.4,
                 loss: str = "BCE",
                 aux: bool = False,
                 **kwargs):
        super(AE, self).__init__()
        self.sparsify = sparsify
        self.noise_std = noise_std
        self.reg_factor = reg_factor
        self.missing_val = missing_val
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.in_dim = in_dim
        self.loss = loss
        self.aux = aux

        print(f"%%% building AE with {encoder} and {decoder}%%%")
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
        x = x.view(-1,1,self.in_dim)
        x = self.encoder(x)
        mu_logvar = self.fc_latent(x)
        mu, _ = mu_logvar.view(-1,self.latent_dim, 2).unbind(-1)
        return mu,_
        # for imputation evaluation
        # return mu, msk

    def decode(self,z):
        return self.decoder(z)

    def reparameterize(self, mu, std):
        # for imputation evaluation
        # return mu
        if self.training:
            eps = torch.randn_like(mu)
            return eps * std + mu
        else:
    #         add no noise for eval mode
            return mu

    def forward(self, input: Tensor) -> Tensor:
        if self.aux:
            aux_label = input[1]
            input = input[0]
        input=input.type(torch.float32)
        # print("$$$ Before encode:", (torch.cuda.memory_allocated(0))/1e6, "MB")
        # for imputation evaluation
        # mu, msk = self.encode(input)
        mu, _ = self.encode(input)
        # print("$$$ After encode:", (torch.cuda.memory_allocated(0))/1e6, "MB")
        # add noise
        z = self.reparameterize(mu, self.noise_std)
        recons = self.decode(z)
        # print("$$$After decode:", (torch.cuda.memory_allocated(0))/1e6, "MB")

        if self.aux:
            aux_logit = self.aux_net(z)
            return [recons , input, z, _, aux_label, aux_logit]
        return [recons , input, z, _]

    def loss_function(self, *args, **kwargs) -> Tensor:
        recons = args[0]
        y_true =args[1]
        z = args[2]
        recons_loss = reconstruction_loss(recons,y_true,self.loss)
        reg_loss = regulization_loss(z,self.reg_factor)
        loss = recons_loss + reg_loss
        if self.aux:
            aux_label = args[4]
            aux_logit = args[5]
            a_loss = aux_loss(aux_logit,aux_label)
            loss += a_loss * 0.0001
        return {'loss': loss, 'Reconstruction_loss': recons_loss.detach(), 'Reg_loss':reg_loss.detach()}