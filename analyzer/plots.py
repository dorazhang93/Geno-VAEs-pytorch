import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from config import PROJECT_DIR

sns.set(font_scale=1.25)

model_name = "best"
plot_out = PROJECT_DIR / "results" / "TPS_eurasia_3899inds_defaultHP_decayLR" / f"{model_name}_ckpt"
plot_out.mkdir(exist_ok=True, parents=True)

def  plot_genotype_hist(genotypes, filename):
	'''
	Plots a histogram of all genotype values in the flattened genotype matrix.

	:param genotypes: array of genotypes
	:param filename: filename (including path) to save plot to
	'''
	unique, counts = np.unique(genotypes, return_counts=True)
	d = zip(unique, counts)
	plt.hist(np.ndarray.flatten(genotypes), bins=50)
	if len(unique) < 5:
		plt.title(", ".join(["{:.2f} : {}".format(u, c) for (u,c) in d]), fontdict = {'fontsize' : 9})

	plt.savefig("{0}.pdf".format(filename))
	plt.close()
def plot_scatters_2fctrs(data,x,y,factors,out,title):
	sns.scatterplot(data=data,x=x,y=y,hue=factors[0],style=factors[1])
	plt.legend()
	plt.title(title)
	plt.savefig(plot_out / out)
def plot_box(data, x, y, out, title):
	sns.boxplot(data=data,x=x,y=y)
	plt.title(title)
	plt.savefig(out)

def plot_paired_heatmap(data1,data2,subtitles,out,title,fmt,vmin,vmax):
	fig, axs = plt.subplots(1,2,figsize=(14,8))
	fig.suptitle(title)
	sns.heatmap(ax=axs[0],data=data1,annot=True,fmt=fmt,vmin=vmin,vmax=vmax)
	sns.heatmap(ax=axs[1],data=data2,annot=True,fmt=fmt,vmin=vmin,vmax=vmax)
	axs[0].set_title(subtitles[0])
	axs[1].set_title(subtitles[1])
	plt.savefig(out)


results = pd.read_csv(PROJECT_DIR / "results" / f"TPS_eurasia_3899inds_defaultHP_decayLR_{model_name}.csv")
results = results.replace("conv", "mlp")

##### CE vs BCE, other factors: model latent
# df_CEvsBCE = results[results["backbone"]=="mlp"]
# plot_box(df_CEvsBCE,x="loss",y="date_mae",out= plot_out / "CEvsBCE.png",title="CrossEntropy vs BinaryCrossEntrioy")

###### factors: backbone, latent, model
df= results[results["loss"]=="BCE"][results["backbone"]!="tfer"]
# predictor: date_mae
df_ae = df[df["model"]=="ae"].pivot("backbone","latent","date_mae")
df_vae = df[df["model"]=="vae"].pivot("backbone","latent","date_mae")
plot_paired_heatmap(df_ae,df_vae,
					subtitles=["ae","vae"],
					out=plot_out / "date_mae_heatmap.png",
					title= "date mean absolute error",
					fmt=".0f",
					vmin=500,
					vmax=1400)
# predictor: location
df_ae = df[df["model"]=="ae"].pivot("backbone","latent","locat_r2")
df_vae = df[df["model"]=="vae"].pivot("backbone","latent","locat_r2")
plot_paired_heatmap(df_ae,df_vae,
					subtitles=["ae","vae"],
					out=plot_out / "location_r2_heatmap.png",
					title= "location r2 ",
					fmt=".4f",
					vmin=0.4,
					vmax=0.9)

# # predictor: impute_error
# df_ae = df[df["model"]=="ae"].pivot("backbone","latent","impute_error")
# df_vae = df[df["model"]=="vae"].pivot("backbone","latent","impute_error")
# plot_paired_heatmap(df_ae,df_vae,
# 					subtitles=["ae","vae"],
# 					out=plot_out / "imputation_error_heatmap.png",
# 					title= "imputation error ",
# 					fmt=".4f",
# 					vmin=0.14,
# 					vmax=0.2)
# # backbone