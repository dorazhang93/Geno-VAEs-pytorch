import torch.nn as nn
from models.module.layers import ResidualBlock, ConvBlock, AttentionPool, MultiheadAttention, Residual, NormConvBlock, GELU
from models.types_ import *
from models.module.helpers import exponential_linspace_int
from einops.layers.torch import Rearrange
import numpy as np
import torch

class encoder(nn.Module):
    def __init__(self):
        pass
    def forward(self):
        pass
# 90M, 5G
class ConvEncoder(nn.Module):
    """
    Convolutional autoencoder for genotype data, as described in [1] Kristiina Ausmees, Carl Nettelblad, A deep learning
     framework for characterization of genotype data, G3 Genes|Genomes|Genetics, 2022;, jkac020,
      https://doi.org/10.1093/g3journal/jkac020
    """
    def __init__(self,
                 in_channel: int,
                 in_dim: int,
                 hidden_dim: int,
                 **kwargs):
        super(ConvEncoder,self).__init__()
        # Build Encoder
        self.encoder = nn.Sequential(
            ConvBlock(in_channel,out_channel=8,kernel_size=5),
            ResidualBlock(in_channel=8, out_channel=8,kernel_size=5,num_layer=2),
            nn.MaxPool1d(kernel_size=5,stride=2,padding=2),
            ConvBlock(8,out_channel=8,kernel_size=5),
            nn.Flatten(),
            nn.Dropout(p=0.01),
            nn.Linear(in_dim*4, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=0.01),
            nn.Linear(hidden_dim,hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
        )

    def forward(self, input: Tensor):
        return self.encoder(input)

# ?G, 21G
class Enformer(nn.Module):
    """
    "Effective gene expression prediction from sequence by integrating long-range interactions"
    Žiga Avsec, Vikram Agarwal, Daniel Visentin, Joseph R. Ledsam, Agnieszka Grabska-Barwinska, Kyle R. Taylor, Yannis Assael, John Jumper, Pushmeet Kohli, David R. Kelley
    https://github.com/deepmind/deepmind-research/tree/master/enformer
    """

    def __init__(self,
                 in_channel: int,
                 in_dim: int,
                 hidden_dim: int,
                 d_token: int,
                 factor_token: int,
                 conv_layer: int,
                 mha_layer: int,
                 heads: int,
                 attn_dropout: float,
                 pos_dropout: float,
                 dropout_rate: float,
                 **kwargs):
        super(Enformer,self).__init__()
        self.d_token = d_token
        self.in_dim = in_dim
        quarter_token = d_token // factor_token

        # create stem
        self.stem = nn.Sequential(
            nn.Conv1d(in_channel, quarter_token, 45, padding=22, stride=2),
            # nn.Conv1d(in_channel, quarter_token, 15, padding=7, stride=2),
            # Residual(NormConvBlock(quarter_token)),
            # AttentionPool(quarter_token, pool_size=2)
        )
        # create conv tower

        filter_list = exponential_linspace_int(quarter_token, d_token, num=conv_layer,
                                               divisible_by=quarter_token)
        filter_list = [quarter_token, *filter_list]

        conv_layers = []
        for dim_in, dim_out in zip(filter_list[:-1], filter_list[1:]):
            conv_layers.append(nn.Sequential(
                NormConvBlock(dim_in, dim_out, kernel_size=15,stride=2),
                # NormConvBlock(dim_in, dim_out, kernel_size=5),
                # Residual(NormConvBlock(dim_out, dim_out, 1)),
                # AttentionPool(dim_out, pool_size=2)
            ))

        self.conv_tower = nn.Sequential(*conv_layers)
        # transformer
        head_list = exponential_linspace_int(heads,2,mha_layer,2)
        transformer = []
        for i in range(mha_layer):
            # TODO reduce dim by 2
            # dim = d_token // (2**i)
            dim = d_token
            head = head_list[i]
            transformer.append(nn.Sequential(
                Residual(nn.Sequential(
                    nn.LayerNorm(dim),
                    MultiheadAttention(
                        dim,
                        heads=head,
                        dim_key=dim // head,
                        dim_value=dim // head,
                        dropout=attn_dropout,
                        pos_dropout=pos_dropout,
                        num_rel_pos_features=dim // head
                    ),
                    nn.Dropout(dropout_rate)
                )),
                nn.Sequential(
                    Rearrange('b n d -> b d n'),
                    # NormConvBlock(dim, dim // 2, kernel_size=5),
                    # AttentionPool(dim // 2, pool_size=2),
                    NormConvBlock(dim, dim, kernel_size=5),
                    AttentionPool(dim, pool_size=2),
                    Rearrange('b d n -> b n d'),
                )
            ))
        self.transformer = nn.Sequential(*transformer)
        # todo reduce dim by 2
        # d_dim = self.d_token // (2**mha_layer)
        d_dim = self.d_token
        l_dim = np.ceil(self.in_dim / (2**(conv_layer+mha_layer+1))).astype(int)

        self._trunk = nn.Sequential(
            self.stem,
            self.conv_tower,
            Rearrange('b d n -> b n d'),
            self.transformer,
        )
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_dim * l_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            GELU(),
        )
    def forward(self,x):
        x = self._trunk(x)
        # print("$$$After encoder _trunk:", (torch.cuda.memory_allocated(0))/1e6, "MB")
        x = self.fc_layer(x)
        # print("$$$After encoder fc_layer:", (torch.cuda.memory_allocated(0))/1e6, "MB")

        return x

# 56M, 15G
class EnBasenjiDiConv(nn.Module):
    def __init__(self,
                 in_channel: int,
                 in_dim: int,
                 hidden_dim: int,
                 d_token: int,
                 factor_token: int,
                 conv_layer: int,
                 di_layer: int,
                 reduce_layer: int,
                 init_dilate: float,
                 dilate_rate_factor: float,
                 dilate_dropout_rate: float,
                 dropout_rate: float,
                 **kwargs):
        super(EnBasenjiDiConv,self).__init__()
        self.d_token = d_token
        self.in_dim = in_dim
        quarter_token = d_token // factor_token

        # create stem
        self.stem = nn.Sequential(
            nn.Conv1d(in_channel, quarter_token, 45, padding=22, stride=2),
        )
        # create conv tower

        filter_list = exponential_linspace_int(quarter_token, d_token, num=conv_layer,
                                               divisible_by=quarter_token)
        filter_list = [quarter_token, *filter_list]

        conv_layers = []
        for dim_in, dim_out in zip(filter_list[:-1], filter_list[1:]):
            conv_layers.append(nn.Sequential(
                NormConvBlock(dim_in, dim_out, kernel_size=15,stride=2),
            ))

        self.conv_tower = nn.Sequential(*conv_layers)
        # Dilated residual layers
        dilate_residuals = []
        token_list = exponential_linspace_int(d_token,d_token*factor_token, num=reduce_layer,divisible_by=16)
        reduce_list = exponential_linspace_int(2,di_layer-1,num=reduce_layer,divisible_by=2)
        assert len(token_list) == len(reduce_list)
        d_dim =d_token
        j = 0
        for i in range(di_layer):
            dilate = int(np.round(init_dilate * (dilate_rate_factor**i)))
            dilate_residuals.append(nn.Sequential(
                Residual(nn.Sequential(
                    NormConvBlock(d_dim,d_dim,kernel_size=3,dilate=dilate,padding="same"),
                    NormConvBlock(d_dim,d_dim,kernel_size=1),
                    nn.Dropout(dilate_dropout_rate)
                )),
            ))
            if i in reduce_list:
                dilate_residuals.append(nn.Sequential(
                    NormConvBlock(d_dim, token_list[j], kernel_size=3),
                    AttentionPool(token_list[j],pool_size=2)
                ))
                d_dim = token_list[j]
                j+=1
        self.dilate_residuals = nn.Sequential(*dilate_residuals)

        d_dim = token_list[-1]
        l_dim = np.ceil(self.in_dim / (2**(conv_layer+reduce_layer+1))).astype(int)

        self._trunk = nn.Sequential(
            self.stem,
            self.conv_tower,
            self.dilate_residuals,
        )
        self.fc_layer = nn.Sequential(
            NormConvBlock(d_dim, d_dim, kernel_size=1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_dim * l_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            GELU(),
        )
    def forward(self,x):
        x = self._trunk(x)
        # print("$$$After encoder _trunk:", (torch.cuda.memory_allocated(0))/1e6, "MB")
        x = self.fc_layer(x)
        # print("$$$After encoder fc_layer:", (torch.cuda.memory_allocated(0))/1e6, "MB")

        return x

# 48M, 12G
class EnBasenjiMlp(nn.Module):
    def __init__(self,
                 in_channel: int,
                 in_dim: int,
                 hidden_dim: int,
                 d_token: int,
                 factor_token: int,
                 conv_layer: int,
                 dropout_rate: float,
                 **kwargs):
        super(EnBasenjiMlp,self).__init__()
        self.d_token = d_token
        self.in_dim = in_dim
        quarter_token = d_token // factor_token

        # create stem
        self.stem = nn.Sequential(
            nn.Conv1d(in_channel, quarter_token, 45, padding=22, stride=2),
        )
        # create conv tower

        filter_list = exponential_linspace_int(quarter_token, d_token, num=conv_layer,
                                               divisible_by=quarter_token)
        filter_list = [quarter_token, *filter_list]

        conv_layers = []
        for dim_in, dim_out in zip(filter_list[:-1], filter_list[1:]):
            conv_layers.append(nn.Sequential(
                NormConvBlock(dim_in, dim_out, kernel_size=15,stride=2),
            ))

        self.conv_tower = nn.Sequential(*conv_layers)
        # MLP layers
        d_dim = self.d_token
        l_dim = np.ceil(self.in_dim / (2**(conv_layer+1))).astype(int)
        self.mlp = nn.Sequential(
            NormConvBlock(d_token,d_token,kernel_size=5),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_dim * l_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim,hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            GELU(),
        )

        self._trunk = nn.Sequential(
            self.stem,
            self.conv_tower,
            self.mlp,
        )
    def forward(self,x):
        x = self._trunk(x)
        # print("$$$After encoder _trunk:", (torch.cuda.memory_allocated(0))/1e6, "MB")
        return x
