import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import glob, sys, os
from plot import errors_ae_don, violin_plots

### Define inputs
latent_dim = [9,25,49,64]
method = ['vanilla-AE','MLAE','CAE']
n_epochs_ae, n_epochs_don = [3000,3000,5000,300], [18000,18000,10000,300]
#latent_dim = [64]
#method = ['CAE']
#n_epochs_ae, n_epochs_don = [5000],[10000]
bs_ae, bs_don = 64,20
ood, noise = 0,0
reps = 5

### Run code for all methods and all latent dimensions
#for j in range(len(latent_dim)):
#    for k in range(len(method)):
#
#        print('Running ' + str(method[k]) + '...') 
#        for i in range(reps):
#            os.system("python AE.py --method " + str(method[k]) + " --latent_dim " + str(latent_dim[j]) + " --n_samples " + str(800) + " --n_epochs " + str(n_epochs_ae[k]) + " --bs " + str(bs_ae) + " --ood " + str(ood) + " --noise " + str(noise))   
#            os.system("python DON.py --method " + str(method[k]) + " --latent_dim " + str(latent_dim[j]) + " --n_samples " + str(800) + " --n_epochs " + str(n_epochs_don[k]) + " --bs " + str(bs_don) + " --ood " + str(ood) + " --noise " + str(noise))  
#        os.system("rm -r results/d_" + str(latent_dim[j]) + "/" + str(method[k]) + "/data/")

### Create final plots
pd_df_AE, pd_df_DON, pd_df_OOD, pd_df_NOISY, _ , _ = errors_ae_don(latent_dim=latent_dim, method=method, reps=reps, ood=ood, noise=noise)
violin_plots(method=method, latent_dim=latent_dim, pd_df_AE=pd_df_AE, pd_df_DON=pd_df_DON, pd_df_OOD=pd_df_OOD, pd_df_NOISY=pd_df_NOISY, save=True)

