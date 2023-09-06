import torch.nn as nn
from models.types_ import *
from models.module.layers import ConvBlock, ResidualBlock,GELU, Residual, MultiheadAttention, NormConvBlock,TargetLengthCrop
import numpy as np
from models.module.helpers import exponential_linspace_int
from einops.layers.torch import Rearrange
import torch

class decoder(nn.Module):
    def __init__(self):
        pass
    def forward(self):
        pass

class View(nn.Module):
    def __init__(self, dim,  shape):
        super(View, self).__init__()
        self.dim = dim
        self.shape = shape

    def forward(self, input):
        new_shape = list(input.shape)[:self.dim] + list(self.shape) + list(input.shape)[self.dim+1:]
        return input.view(*new_shape)

class ConvDecoder(nn.Module):
    def __init__(self,
                 latent_dim: int,
                 hidden_dim: int,
                 out_channel: int,
                 out_dim: int,
                 **kwargs):
        super(ConvDecoder, self).__init__()
        self.out_dim = out_dim
        # Build decoder
        self.decoder1 = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Dropout(0.01),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Dropout(0.01),
            nn.Linear(hidden_dim, out_dim*4),
            nn.BatchNorm1d(out_dim*4),
            View(1,(8,int(out_dim/2))),
        )
        self.decoder2 = nn.Sequential(
            ConvBlock(8,8,5),
            nn.Upsample(scale_factor=2),
            ResidualBlock(in_channel=8,out_channel=8,kernel_size=5,num_layer=2),
            nn.Conv1d(in_channels=8,out_channels=8,kernel_size=5,padding=2),
            nn.ELU()
        )
        self.decoder3 = nn.Sequential(
            nn.BatchNorm1d(8),
            nn.Conv1d(in_channels=8,out_channels=out_channel,kernel_size=1),
            nn.Flatten(),
        )
    def forward(self, z):
        x = self.decoder1(z)
        x = self.decoder2(x)
        x = self.decoder3(x)

        return x

# TODO insert convblock between upsample and dropout
class Deformer(nn.Module):
    def __init__(self,
                 latent_dim: int,
                 hidden_dim: int,
                 out_channel: int,
                 out_dim: int,
                 d_token: int,
                 factor_token: int,
                 conv_layer: int,
                 mha_layer: int,
                 heads: int,
                 attn_dropout: float,
                 pos_dropout: float,
                 dropout_rate: float,
                 **kwargs):
        super(Deformer, self).__init__()
        self.out_dim = out_dim
        quarter_token = d_token // factor_token
        self.out_channel = out_channel
        # d_dim = d_token // (2**mha_layer)
        d_dim = d_token
        l_dim = np.ceil(self.out_dim / (2**(conv_layer+mha_layer+1))).astype(int)
        self.dense_layer = nn.Sequential(
            nn.Linear(latent_dim,hidden_dim),
            GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, d_dim*l_dim),
            nn.BatchNorm1d(d_dim*l_dim),
            View(1,(l_dim,d_dim)),
        )
        #     (b n d)
        #     transformer
        head_list = exponential_linspace_int(2,heads,mha_layer,2)
        transformer = []
        for i in range(mha_layer):
            # dim = d_dim * (2**i)
            # new_dim = dim * 2
            dim = d_token
            new_dim = d_token
            head = head_list[i]
            transformer.append(nn.Sequential(
                nn.Sequential(
                    Rearrange('b n d -> b d n'),
                    nn.Upsample(scale_factor=2),
                    NormConvBlock(dim, new_dim, kernel_size=5),
                    Rearrange('b d n -> b n d'),
                ),
            Residual(nn.Sequential(
                    nn.LayerNorm(new_dim),
                    MultiheadAttention(
                        new_dim,
                        heads=head,
                        dim_key=new_dim // head,
                        dim_value=new_dim // head,
                        dropout=attn_dropout,
                        pos_dropout=pos_dropout,
                        num_rel_pos_features= new_dim // head
                    ),
                    nn.Dropout(dropout_rate)
                )),
            ))
        self.transformer = nn.Sequential(*transformer)
    #     conv tower
        filter_list = exponential_linspace_int(d_token, quarter_token, num=conv_layer,
                                               divisible_by=quarter_token)
        filter_list = [d_token, *filter_list]

        conv_layers = []
        for dim_in, dim_out in zip(filter_list[:-1], filter_list[1:]):
            conv_layers.append(nn.Sequential(
                nn.Upsample(scale_factor=2),
                NormConvBlock(dim_in, dim_out, kernel_size=15),
                # Residual(NormConvBlock(dim_in, dim_in, 1)),
                # NormConvBlock(dim_in, dim_out, kernel_size=5),
            ))
        self.conv_tower = nn.Sequential(*conv_layers)

        # (b,d,n)
        target_length = self.out_dim // 2

        self.crop = TargetLengthCrop(target_length)
        self.stem = nn.Sequential(
            nn.Upsample(scale_factor=2),
            # Residual(NormConvBlock(quarter_token)),
            # nn.Conv1d(quarter_token,self.out_channel, 15, padding=7),
            nn.Conv1d(quarter_token,self.out_channel, 45, padding=22),
        )
        self._reversed_trunk = nn.Sequential(
            self.transformer,
            Rearrange('b n d -> b d n'),
            self.conv_tower,
            self.crop,
        )
        self._final =nn.Sequential(
            self.stem,
            nn.Flatten(),
        )


    def forward(self, z):
        # print("$$$ Before decoder forward:", (torch.cuda.memory_allocated(0))/1e6, "MB")
        x = self.dense_layer(z)
        # print("$$$ After decoder dense_layer:", (torch.cuda.memory_allocated(0))/1e6, "MB")
        x = self._reversed_trunk(x)
        # print("$$$ After decoder _reversed_trunk:", (torch.cuda.memory_allocated(0))/1e6, "MB")
        x = self._final(x)
        # print("$$$ After decoder _final:", (torch.cuda.memory_allocated(0))/1e6, "MB")
        return x

class DeBasenjiDiConv(nn.Module):
    def __init__(self,
                 latent_dim: int,
                 hidden_dim: int,
                 out_channel: int,
                 out_dim: int,
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
        super(DeBasenjiDiConv, self).__init__()
        self.out_dim = out_dim
        quarter_token = d_token // factor_token
        self.out_channel = out_channel

        d_dim = d_token * factor_token
        l_dim = np.ceil(self.out_dim / (2**(conv_layer+reduce_layer+1))).astype(int)
        self.dense_layer = nn.Sequential(
            nn.Linear(latent_dim,hidden_dim),
            GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, d_dim*l_dim),
            View(1,(d_dim,l_dim)),
        )
        #     (b d n)
        # Dilated residual layers
        reduce_list = exponential_linspace_int(1,di_layer-1,num=reduce_layer,divisible_by=2)
        token_list = exponential_linspace_int(d_dim, d_token, num=reduce_layer,divisible_by=16)
        assert len(token_list) == len(reduce_list)
        dilate_residuals = []
        j=0
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
                    NormConvBlock(d_dim,token_list[j],kernel_size=3),
                    nn.Upsample(scale_factor=2),
                ))
                d_dim = token_list[j]
                j+=1
        self.dilate_residuals = nn.Sequential(*dilate_residuals)
    #     conv tower
        filter_list = exponential_linspace_int(d_token, quarter_token, num=conv_layer,
                                               divisible_by=quarter_token)
        filter_list = [d_token, *filter_list]

        conv_layers = []
        for dim_in, dim_out in zip(filter_list[:-1], filter_list[1:]):
            conv_layers.append(nn.Sequential(
                NormConvBlock(dim_in, dim_out, kernel_size=15),
                nn.Upsample(scale_factor=2),
                # Residual(NormConvBlock(dim_in, dim_in, 1)),
                # NormConvBlock(dim_in, dim_out, kernel_size=5),
            ))
        self.conv_tower = nn.Sequential(*conv_layers)

        # (b,d,n)
        target_length = self.out_dim // 2

        self.crop = TargetLengthCrop(target_length)
        self.stem = nn.Sequential(
            NormConvBlock(quarter_token,quarter_token,kernel_size=5),
            nn.Upsample(scale_factor=2),
            # Residual(NormConvBlock(quarter_token)),
            # nn.Conv1d(quarter_token,self.out_channel, 15, padding=7),
            nn.Conv1d(quarter_token,self.out_channel, 45, padding=22),
        )
        self._reversed_trunk = nn.Sequential(
            self.dilate_residuals,
            self.conv_tower,
            self.crop,
        )
        self._final =nn.Sequential(
            self.stem,
            nn.Flatten(),
        )


    def forward(self, z):
        # print("$$$ Before decoder forward:", (torch.cuda.memory_allocated(0))/1e6, "MB")
        x = self.dense_layer(z)
        # print("$$$ After decoder dense_layer:", (torch.cuda.memory_allocated(0))/1e6, "MB")
        x = self._reversed_trunk(x)
        # print("$$$ After decoder _reversed_trunk:", (torch.cuda.memory_allocated(0))/1e6, "MB")
        x = self._final(x)
        # print("$$$ After decoder _final:", (torch.cuda.memory_allocated(0))/1e6, "MB")
        return x

class DeBasenjiMlp(nn.Module):
    def __init__(self,
                 latent_dim: int,
                 hidden_dim: int,
                 out_channel: int,
                 out_dim: int,
                 d_token: int,
                 factor_token: int,
                 conv_layer: int,
                 dropout_rate: float,
                 **kwargs):
        super(DeBasenjiMlp, self).__init__()
        self.out_dim = out_dim
        quarter_token = d_token // factor_token
        self.out_channel = out_channel
        d_dim = d_token
        l_dim = np.ceil(self.out_dim / (2**(conv_layer+1))).astype(int)
        #     MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim,hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim,hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, d_dim*l_dim),
            nn.BatchNorm1d(d_dim*l_dim),
            View(1,(d_dim,l_dim)),
        )
        #     (b d n)
    #     conv tower
        filter_list = exponential_linspace_int(d_token, quarter_token, num=conv_layer,
                                               divisible_by=quarter_token)
        filter_list = [d_token, *filter_list]

        conv_layers = []
        for dim_in, dim_out in zip(filter_list[:-1], filter_list[1:]):
            conv_layers.append(nn.Sequential(
                nn.Upsample(scale_factor=2),
                NormConvBlock(dim_in, dim_out, kernel_size=15),
            ))
        self.conv_tower = nn.Sequential(*conv_layers)

        # (b,d,n)
        target_length = self.out_dim // 2

        self.crop = TargetLengthCrop(target_length)
        self.stem = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(quarter_token,self.out_channel, 45, padding=22),
        )
        self._reversed_trunk = nn.Sequential(
            self.mlp,
            self.conv_tower,
            self.crop,
        )
        self._final =nn.Sequential(
            self.stem,
            nn.Flatten(),
        )


    def forward(self, z):
        x = self._reversed_trunk(z)
        # print("$$$ After decoder _reversed_trunk:", (torch.cuda.memory_allocated(0))/1e6, "MB")
        x = self._final(x)
        # print("$$$ After decoder _final:", (torch.cuda.memory_allocated(0))/1e6, "MB")
        return x

