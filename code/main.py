import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import glob, sys, os
from plot import errors_ae_don, violin_plots

### Define inputs
latent_dim = [36]
method = ['WAE']
n_epochs_ae, n_epochs_don = [3], [3]
bs_ae, bs_don = 64,20
ood, noise = 1,1
reps = 5

### Run code for all methods and all latent dimensions
for j in range(len(latent_dim)):
    for k in range(len(method)):

        print('Running ' + str(method[k]) + '...') 
        for i in range(reps):
            os.system("python AE.py --method " + str(method[k]) + " --latent_dim " + str(latent_dim[j]) + " --n_samples " + str(800) + " --n_epochs " + str(n_epochs_ae[j]) + " --bs " + str(bs_ae) + " --ood " + str(ood) + " --noise " + str(noise))   
            os.system("python DON.py --method " + str(method[k]) + " --latent_dim " + str(latent_dim[j]) + " --n_samples " + str(800) + " --n_epochs " + str(n_epochs_don[j]) + " --bs " + str(bs_don) + " --ood " + str(ood) + " --noise " + str(noise))  

### Create final plots
pd_df_AE, pd_df_DON, pd_df_OOD, pd_df_NOISY, _ , _ = errors_ae_don(latent_dim=latent_dim, method=method, reps=reps)
violin_plots(method=method, latent_dim=latent_dim, pd_df_AE=pd_df_AE, pd_df_DON=pd_df_DON, pd_df_OOD=pd_df_OOD, pd_df_NOISY=pd_df_NOISY, save=True)
