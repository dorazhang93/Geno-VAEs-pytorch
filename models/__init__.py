from .base import *
from .geno_cae import *
from .ae import AE
from .vae import VAE
from .i_vae import iVAE
from .id_vae import idVAE

vae_models = {'GenoCAE':GenoCAE,
              'AutoEncoder': AE,
              "VAE": VAE,
              "iVAE": iVAE,
              "idVAE":idVAE}