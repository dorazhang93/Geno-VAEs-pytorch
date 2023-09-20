import json, os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from typing import List, Optional, Sequence, Union, Any, Callable
from pytorch_lightning import LightningDataModule
from scipy import stats


class DNA(Dataset):
    def __init__(self,
                 data_dir: str,
                 split :str = "train",
                 transform: Callable =None,
                 MFA: bool = False,
                 aux: bool = False,
                 **kwargs):
        self.data_dir = data_dir
        self.transforms = transform
        self.aux = aux

        self.inds = np.load(self.data_dir+f"_MF_inds_{split}.npy")
        self.genotypes = np.load(self.data_dir+f"_MF_genotype_{split}.npy")
        print("@"*10+split+"@"*10)
        print(f"Load data from {self.data_dir}")
        print("genotype data shape:", self.genotypes.shape)
        print("inds shape", self.inds.shape)
        self.mf = None
        if MFA:
            self.masks = np.load(self.data_dir + f"_MF_mask_{split}.npy")
            print("mask shape", self.masks.shape)
            self.mf = self._get_most_frequent()
        self.pops = None
        if self.aux:
            if os.path.exists(self.data_dir + f"_continent_oh_{split}.npy"):
                self.pops = np.load(self.data_dir + f"_continent_oh_{split}.npy")
            else:
                self.pops = self._encode_pop()
                np.save(self.data_dir + f"_continent_oh_{split}.npy", self.pops)
            print("pop labels shape", self.pops.shape)

    def _encode_pop(self):
        with open("/home/daqu/Projects/data/aDNA/TPS_global_country_continent.json","r") as f:
            country_continent_map = json.loads(f.read())
        continents = [country_continent_map[x] if x!="Rare_pop" else "Unknown" for x in self.inds[:,1]]
        num_contients = len(list(set(continents)-set(["Unknown"])))
        continent_emb = {"Europe":0, "Central_and_Western_Asia":1, "Eastern_Asia_and_Oceania":2,
                        "Africa":3, "America":4,"Unknown":-1}
        def one_hot(x):
            e = np.zeros(num_contients)
            # rare population
            if x == -1:
                return e
            else:
                assert x<num_contients
                e[x] =1
            return e

        return np.array([one_hot(continent_emb[x]) for x in continents])

    def _get_most_frequent(self,):
        print("get most frequent allele for each SNP")
        geno = np.load(self.data_dir+"_MF_genotype_all.npy")
        msk = np.load(self.data_dir+"_MF_mask_all.npy")
        print("geno shape:" , geno.shape)
        print("mask shape:", msk.shape)
        mf = np.empty(geno.shape[1])
        for i in range(geno.shape[1]):
            mf_impute = geno[:,i][(1-msk[:,i]).astype(bool)]
            if mf_impute.size>0:
                mf[i]=mf_impute[0]
            else:
                mf[i]= stats.mode(geno[:,i]).mode[0]
        return mf


    def __len__(self):
        assert len(self.genotypes)==len(self.inds)
        return len(self.genotypes)

    def __getitem__(self, idx):
        input = self.genotypes[idx]
        if self.aux:
            sup_pop = self.pops[idx]
            return torch.from_numpy(input), torch.from_numpy(sup_pop)
        else:
            return torch.from_numpy(input)

class VAEDataset(LightningDataModule):
    def __init__(self,
                 data_path: str,
                 data_name: str,
                 train_batch_size: int = 64,
                 val_batch_size: int = 64,
                 num_workers: int = 0,
                 pin_memory: bool =False,
                 aux_on: bool = False,
                 **kwargs):
        super().__init__()

        self.data_dir = data_path
        self.data_name = data_name
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.aux_on = aux_on
    def prepare_data(self):
        pass
    def setup(self,stage=None):
        # for cross validation
        # if stage=="predict":
        #     self.all_dataset = DNA(
        #     self.data_dir + self.data_name,
        #     split="all",
        #     )
        #     return
        # for external test
        if stage=="predict":
            self.train_dataset = DNA(
            "/home/daqu/Projects/data/aDNA/Eurasian/TPS_6535inds/TPS_eurasia_train6535",
            split="all",
            aux = self.aux_on
            )
            self.test_dataset = DNA(
            "/home/daqu/Projects/data/aDNA/Eurasian/TPS_test1264/TPS_eurasia_test1264",
            split="all",
            aux = self.aux_on
            )
            return
        elif stage=="fit":
            self.train_dataset = DNA(
            self.data_dir+self.data_name,
            split="train",
            aux = self.aux_on
            )
            self.val_dataset = DNA(
            self.data_dir + self.data_name,
            split="val",
            aux = self.aux_on
            )
        elif stage=="test":
            self.val_dataset = DNA(
            self.data_dir + self.data_name,
            split="val",
            MFA=True,
            aux = self.aux_on
            )
        else:
            raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size= self.train_batch_size,
            num_workers= self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory
        )

    def all_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory
        ),\
            DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory
        )

    # for cross validation
    # def all_dataloader(self):
    #     return DataLoader(
    #         self.all_dataset,
    #         batch_size=self.val_batch_size,
    #         num_workers=self.num_workers,
    #         shuffle=False,
    #         pin_memory=self.pin_memory
    #     )
