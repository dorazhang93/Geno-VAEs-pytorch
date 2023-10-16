import pandas as pd
import json
import yaml
from config import PROJECT_DIR
import os

DATASETS = ["TPS_eurasia_train3899"]
MODELS = ["ae", "vae"]
LOSS = ["BCE"]
BACKBONE = ["conv"]
LATENT = [5, 10, 15]
model_name = "best"
out_file = f"TPS_eurasia_3899inds_stageIEn6k_stageIIEn6kEurasia_{model_name}.csv"
# ENCODE="EnEurasia"
# ENCODE="testEn1k"
# ENCODE="EnGlobal"
ENCODE="stageIIEn6kEurasia"
def get_hparams(out_folder):
    hparams_file = out_folder / 'hparams.yaml'
    if not os.path.exists(hparams_file):
        return None,None
    with open(hparams_file, "r") as f:
        hparams = yaml.safe_load(f)
    return hparams["exp_params"]["LR"], hparams["model_params"]["sparsify"]

def get_project_loss(dataset,experiment):
    project_file = PROJECT_DIR / "log" / f"project_{dataset}_{experiment}.txt"
    if not os.path.exists(project_file):
        return None
    train_loss=None
    test_loss=None
    with open(project_file,"r") as f:
        for line in f.readlines():
            if line.startswith("train Average loss"):
                try:
                    train_loss = float(line.strip("\n").strip("train Average loss :"))
                except:
                    pass
            if line.startswith("test Average loss"):
                try:
                    test_loss = float(line.strip("\n").strip("test Average loss :"))
                except:
                    pass
    return train_loss, test_loss

def get_val_loss(out_folder):
    metrics_file = out_folder / "metrics.json"
    if not os.path.exists(metrics_file):
        return None
    with open(metrics_file, "r") as f:
        metrics = json.load(f)
    return metrics["val"]["Reconstruction_loss"]

def get_Date_error(out_folder):
    file = out_folder / f"Date_regression_{ENCODE}.json"
    if not os.path.exists(file):
        return None,None
    with open(file, "r") as f:
        metrics = json.load(f)
    return metrics["Date regression"]["Mean Absolute Error"]["train"]["average"], \
           metrics["Date regression"]["Mean Absolute Error"]["test"]["average"], \
           metrics["Date regression"]["Mean Absolute Error"]["test"]["std"]

def get_location_error(out_folder):
    file = out_folder / f"Location_regression_{ENCODE}.json"
    if not os.path.exists(file):
        return None
    with open(file, "r") as f:
        metrics = json.load(f)
    return metrics["Location regression"]["r2 score"]["train"]["average"],\
           metrics["Location regression"]["r2 score"]["test"]["average"]

def get_impute_error(out_folder):
    file = out_folder / "impute_error.json"
    if not os.path.exists(file):
        return -1.0,-1.0
    with open(file, "r") as f:
        metrics = json.load(f)
    return metrics["AE_error"][-1],metrics["miss_rates"][-1]


# results = pd.DataFrame([],columns=["data","model","loss","backbone","latent","lr","sparsify","reconst_loss_train",
#                                    "reconst_loss_test","date_mae","date_mae_std","locat_r2","impute_error",
#                                    "miss_rate"])
results = pd.DataFrame([],columns=["data","model","loss","backbone","latent","lr","sparsify","reconst_loss_train",
                                   "reconst_loss_test","date_mae_train","date_mae_test","date_mae_std","locat_r2_train","locat_r2_test"])
for data in DATASETS:
    for model in MODELS:
        for loss in LOSS:
            for backbone in BACKBONE:
                for latent in LATENT:
                    experiment = f"{model}_{loss}_{backbone}_{latent}"
                    if model == "ae":
                        out_folder = PROJECT_DIR / "outputs" / f"{data}" / "AutoEncoder" / f"{experiment}"
                    elif model == "vae":
                        out_folder = PROJECT_DIR / "outputs" / f"{data}" / "VAE" / f"{experiment}"
                    elif model == "ivae":
                        out_folder = PROJECT_DIR / "outputs" / f"{data}" / "iVAE" / f"{experiment}"
                    elif model == "idvae":
                        out_folder = PROJECT_DIR / "outputs" / f"{data}" / "idVAE" / f"{experiment}"
                    elif model == "vaeaux":
                        out_folder = PROJECT_DIR / "outputs" / f"{data}" / "VAE" / f"{experiment}"
                    elif model == "aeaux":
                        out_folder = PROJECT_DIR / "outputs" / f"{data}" / "AutoEncoder" / f"{experiment}"
                    else:
                        raise ValueError(f"invalid model {model}") #["ivae" "idvae" "vaeaux" "aeaux"]
                    if not out_folder.exists():
                        continue
                    lr, sparsify = get_hparams(out_folder)
                    train_loss, test_loss = get_project_loss(data,experiment)
                    # val_loss = get_val_loss(out_folder)
                    date_mae_train, date_mae_test, date_mae_std = get_Date_error(out_folder)
                    locat_r2_train, locat_r2_test = get_location_error(out_folder)
                    # impute_error, miss_rate= get_impute_error(out_folder)
                    results = pd.concat([results, pd.DataFrame({"data": [data],
                                                 "model": [model],
                                                 "loss":[loss],
                                                 "backbone":[backbone],
                                                 "latent":[latent],
                                                 "lr":[lr],
                                                 "sparsify":[sparsify],
                                                 "reconst_loss_train":[train_loss],
                                                 "reconst_loss_test":[test_loss],
                                                 "date_mae_train":[date_mae_train],
                                                 "date_mae_test": [date_mae_test],
                                                 "date_mae_std":[date_mae_std],
                                                 "locat_r2_train":[locat_r2_train],
                                                 "locat_r2_test":[locat_r2_test]
                                                 # "impute_error":[impute_error],
                                                 # "miss_rate":[miss_rate],
                                                           })],
                                   ignore_index=True)

results.to_csv(PROJECT_DIR/ "results"/ out_file,float_format="%.6f")
