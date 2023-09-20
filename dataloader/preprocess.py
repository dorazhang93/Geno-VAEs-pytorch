import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from analyzer.plots import plot_genotype_hist

rng = np.random.RandomState(0)
FULL_FILEBASE = "/home/daqu/Projects/data/aDNA/Eurasian_extension/TPS_global_full"
TEST_FILEBASE = "/home/daqu/Projects/data/aDNA/Eurasian/TPS_test1264/TPS_eurasia_test1264"
def g(file):
    with open(file,"r") as f:
        for line in f.readlines():
            for item in list(map(int, line.strip("\n"))):
                yield item

def compare_snp(snp_profile1,snp_profile2):
    assert len(snp_profile1)==len(snp_profile2)
    flipped_loc=[]
    unkown_loc=[]
    for i in range(len(snp_profile1)):
        snp1=snp_profile1[i]
        snp2=snp_profile2[i]
        if snp2[0] == snp1[0]:
            if snp2[4]==snp1[4] and snp2[5]==snp1[5]:
                continue
            elif snp2[5]==snp1[4] and snp2[4]==snp1[5]:
                flipped_loc.append(i)
            else:
                print(snp1)
                print(snp2)
                unkown_loc.append(i)
        else:
            raise ("Mismatched SNP orders")
    return flipped_loc,unkown_loc

class Preprocesser:
    def __init__(self,
                 filebase: str,
                 normalization: bool = True,
                 impute_missing: str = "MF",
                 train_val_split: float = 0.2,
                 missing_val_ori: float = 9.0,
                 missing_val_norm: float = -1.0,
                 split_policy: str = "random",
                 sanity_check = False):
        """
        @param filebase: path+ filename without suffix eg. "/home/data/TPS_full/new_TPS"
        @param normalization: wheter to normalize input
        @param impute_missing: method for missing value imputation, no imputation if None
        @param train_val_split: fraction of validation set
        @param missing_val_ori: missing value in input
        @param missing_val_norm: missing value after normalization
        """
        self.filebase = filebase
        self.normalization = normalization
        self.impute_missing = impute_missing
        self.split = train_val_split
        self.missing_val_ori = missing_val_ori
        self.missing_val_norm = missing_val_norm
        self.split_policy = split_policy
        if sanity_check:
            print("sanity check ...")
            if self._sanity_check():
                print("Passed sanity check")
            else:
                raise ("Failed sanity check")
        self.inds, self.targets = self._load_inds()
        self.genotypes = self._load_genotype()
        self.missing_mask = (self.genotypes!=self.missing_val_ori).astype(int).T # 0 means missing
        print(f"Overall missing rate is {1-self.missing_mask.sum()/self.missing_mask.size}")

    def _sanity_check(self):
        # load full set
        full_inds = np.genfromtxt(FULL_FILEBASE+".ind", usecols=(0),dtype=str)
        print("Reading individual profiles from " + FULL_FILEBASE + ".ind")
        full_snps = np.genfromtxt(FULL_FILEBASE+".snp",dtype=str)
        print("Reading snps profiles from" + FULL_FILEBASE + ".snp")
        full_geno = np.fromiter(g(FULL_FILEBASE+".eigenstratgeno"),dtype=float).reshape(-1,len(full_inds))
        # load test set
        test_inds = np.genfromtxt(TEST_FILEBASE+".ind", usecols=(0),dtype=str)
        print("Reading individual profiles from " + TEST_FILEBASE + ".ind")
        test_snps = np.genfromtxt(TEST_FILEBASE+".snp",dtype=str)
        print("Reading snps profiles from" + TEST_FILEBASE + ".snp")
        test_geno = np.fromiter(g(TEST_FILEBASE+".eigenstratgeno"),dtype=float).reshape(-1,len(test_inds))
        # compare snp file
        snp_where_flipped, unknown = compare_snp(full_snps,test_snps)
        test_of_full = (full_geno.T)[np.array([np.where(full_inds==x)[0][0] for x in test_inds])]
        test_geno = test_geno.T
        # test_geno = np.load("test.npy")
        # test_of_full = np.load("test_of_full.npy")
        np.save("test.npy",test_geno)
        np.save("test_of_full.npy",test_of_full)
        print(test_of_full.shape, test_geno.shape)
        for idx in snp_where_flipped:
            test_of_full[:,idx]=2-test_of_full[:,idx]
        for idx in unknown:
            test_of_full[:,idx]=test_geno[:,idx]
        test_of_full[test_of_full == -7] = 9
        return np.array_equal(test_of_full,test_geno)

    def _load_inds(self):
        """
        @return: individual and corresponding population
        @rtype: np.array in shape of (num_inds, 2), col1 is ID, col2 is population
        """
        ind_pop_list = np.genfromtxt(self.filebase + ".ind", usecols=(0, 2), dtype=str)
        print("Reading individual profiles from " + self.filebase + ".ind")
        targets = np.genfromtxt(self.filebase + ".ind", usecols=(3,4,5,6), dtype=float)
        print("Reading targets from " + self.filebase + ".ind")
        return ind_pop_list, targets

    def _load_genotype(self):
        """
        @return: raw genotype data in 0,1,2,9(missing)
        @rtype: np.array in shape of (num_SNPs, num_inds)
        """
        # load full set
        full_inds = np.genfromtxt(FULL_FILEBASE+".ind", usecols=(0),dtype=str)
        print("Reading individual profiles from " + FULL_FILEBASE + ".ind")
        full_geno = np.fromiter(g(FULL_FILEBASE+".eigenstratgeno"),dtype=float).reshape(-1,len(full_inds))
        genotypes = (full_geno.T)[np.array([np.where(full_inds==x)[0][0] for x in self.inds[:,0]])]
        print("genotypes's shape", genotypes.shape)
        return genotypes.T
    def _normalize(self):
        """
        Normalize genotypes into interval [0,1] by translating 0,1,2 -> 0.0, 0.5, 1.0, missing value (default 9) -> -1
        """
        self.genotypes[self.genotypes == 1.0] = 0.5
        self.genotypes[self.genotypes == 2.0] = 1.0
        self.genotypes[self.genotypes == self.missing_val_ori] = self.missing_val_norm

    def _impute_missing(self):
        if self.impute_missing == "MF":
            self._most_frequent_impute()
            print("Imputed missing value using most frequent allele")
        elif self.impute_missing == "AE":
            self._AE_impute()
            print("Imputed missing value using AE reconstruction")
        else:
            raise NotImplementedError(f"{self.impute_missing} is not implemented")

    def _most_frequent_impute(self):
        for m in self.genotypes:
            # exclude missing value before count modes
            modes = stats.mode(m[m!=self.missing_val_norm])
            most_frequent = modes.mode
            if most_frequent.size ==0:
                Warning("Run into SNPs with 100% missing rate")
            else:
                m[m==self.missing_val_norm]=most_frequent[0]

    def _AE_impute(self):
        # Reconstructed genotype file should be placed at the same folder as .eigenstratgeno file
        # and with suffix .npy
        reconstruct= np.load(self.filebase+".npy", dtype=float)
        print ("load reconstructed genotype from "+ self.filebase+".npy")
        assert reconstruct.shape == self.genotypes
        # normalize before autoencoder imputation
        if not self.normalization:
            print("Normalization before AE")
            self._normalize()
        self.genotypes = self.genotypes * self.missing_mask + reconstruct * (1-self.missing_mask)


    def _train_val_split(self):
        if self.split_policy == "random":
            geno_train, geno_val, inds_train, inds_val, mask_train, mask_val= train_test_split(self.genotypes, self.inds,
                                                                                               self.missing_mask,
                                                                                               test_size=self.split,
                                                                                               random_state=rng)
        elif self.split_policy == "stratified":
            pop_list=self.inds[:,1]
            pops, pop_counts = np.unique(pop_list,return_counts=True)
            rare_pops = pops[pop_counts==1]
            for rare in rare_pops:
                pop_list[pop_list==rare]="Rare_pop"

            geno_train, geno_val, inds_train, inds_val, mask_train, mask_val = train_test_split(self.genotypes, self.inds,
                                                                                                self.missing_mask,
                                                                                                test_size=self.split,
                                                                                                random_state=rng,
                                                                                                stratify=pop_list)
        else:
            raise NotImplementedError(f"{self.split_policy} is not implemented" )
        return geno_train, geno_val, inds_train, inds_val, mask_train, mask_val

    def _save(self):
        return

    def process(self, split_val=True):
        """
        input should be raw genotype data in 0,1,2,9(missing)
        """
        if self.normalization:
            self._normalize()
            print("Normalized genome by mapping 0 to 0, 1 to 0.5, 2 to 1.0")
        if self.impute_missing is not None:
            self._impute_missing()
        #     transpose genotype data into shape of (n_inds, n_SNPs)
        self.genotypes = self.genotypes.T
        plot_genotype_hist(self.genotypes,self.filebase+"_genotype_hist")

        if split_val:
            geno_train, geno_val, inds_train, inds_val, mask_train, mask_val = self._train_val_split()
            # save data
            data_dict = {"genotype":{"all":self.genotypes,"train":geno_train,"val":geno_val},
                        "inds":{"all":self.inds,"train":inds_train,"val":inds_val},
                        "mask":{"all":self.missing_mask,"train":mask_train,"val":mask_val}}
            for dtyp , v in data_dict.items():
                for split, value in v.items():
                    if dtyp == "genotype":
                        np.save(self.filebase+f"_{self.impute_missing}_{dtyp}_{split}.npy",value.astype(np.float16))
                    elif dtyp == "inds":
                        np.save(self.filebase+f"_{self.impute_missing}_{dtyp}_{split}.npy",value)
                    elif dtyp =="mask":
                        np.save(self.filebase+f"_{self.impute_missing}_{dtyp}_{split}.npy",value.astype(int))
                    else:
                        raise ValueError(f"Wrong data {dtyp}")
            np.save(self.filebase+f"_{self.impute_missing}_Y_all.npy", self.targets)
        else:
            np.save(self.filebase + f"_{self.impute_missing}_genotype_all.npy", self.genotypes.astype(np.float16))
            np.save(self.filebase+f"_{self.impute_missing}_inds_all.npy",self.inds)
            np.save(self.filebase+f"_{self.impute_missing}_Y_all.npy", self.targets)



if __name__ == "__main__":
    params= {"filebase":"/home/daqu/Projects/data/aDNA/Eurasian_extension/TPS_global_mind01_dsM_ddup/TPS_global_mind01_dsM_ddup",
             "normalization": True,
             "impute_missing": "MF",
             "train_val_split": 0.2,
             "missing_val_ori": 9.0,
             "split_policy": "stratified",
             "sanity_check": False}
    processer = Preprocesser(**params)
    processer.process(split_val=True)