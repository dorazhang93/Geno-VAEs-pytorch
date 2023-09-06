from torch.nn import functional as F
import torch
import numpy as np

def alfreqvector(y_pred):
    n,l = y_pred.shape
    alfreq = torch.sigmoid(y_pred).view(n,l,1)
    return torch.cat(((1-alfreq)**2,2*alfreq*(1-alfreq),alfreq**2), dim=2)

def y_onehot(y_true):
    y_true = F.one_hot((y_true * 2).long(), num_classes=3)
    return y_true

def reconstruction_loss(recons, y_true, loss = "CE"):
    if loss == "CE":
        recons = torch.reshape(alfreqvector(recons),(-1,3))
        y_true = torch.reshape((y_true * 2).long(), (-1,))
        recons_loss = F.cross_entropy(recons, y_true, reduction='none')
    elif loss == "BCE":
        recons_loss = F.binary_cross_entropy_with_logits(recons,y_true, reduction='none')
    else:
        raise NotImplementedError
    recons_loss = recons_loss[y_true>=0].mean()
    return recons_loss

def KLD_loss(latent_dist):
    mu , logvar = latent_dist
    kld_loss = torch.mean(-0.5 * torch.mean(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
    return kld_loss

def regulization_loss(z, reg_factor):
    reg_loss = reg_factor * torch.mean(z ** 2)
    return reg_loss

def aux_loss(aux_logit, aux_label):
    return F.cross_entropy(aux_logit,aux_label)

def log_Normal_pdf(z, mu, logv):
    lpdf = -0.5 * (torch.log(2 * np.pi * torch.ones(1).to("cuda:0")) + logv + (z-mu)**2 * torch.exp(-logv))
    return lpdf.mean()

