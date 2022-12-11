import numpy as np
from matplotlib import pylab as plt
import argparse
import os, sys, glob
import pandas as pd
import seaborn as sns

if not os.path.exists('results/final_results/'):
    os.makedirs('results/final_results/') 

dir_final_res = 'results/final_results/'  
    
def errors_ae_don(latent_dim, method, reps, ood=0, noise=0):
    
    ''' Returns dataframes of errors and errors for AE and DON
        latent_dim: list of all latent dimensions
        method: list of all methods 
    '''
    ### Autoencoder
    errors_AE_dict = {}
    for j in range(len(method)):
        error_AE = {}
        for k in range(len(latent_dim)):
            my_files = glob.glob('results/d_' + str(latent_dim[k]) + '/' + str(method[j]) + '/errors_AE/' + '*.txt')
            error_temp = []
            for i in range(reps):
                error_temp.append(np.loadtxt(str(my_files[i])))
            error_AE['d={}'.format(latent_dim[k])] = np.array(error_temp)
        errors_AE_dict[method[j]] = error_AE

    ### DeepONet
    errors_DON_dict = {}
    for j in range(len(method)):
        error_DON = {}
        for k in range(len(latent_dim)):
            my_files = glob.glob('results/d_' + str(latent_dim[k]) + '/' + str(method[j]) + '/errors_DON/' + '*.txt')
            error_temp = []
            for i in range(reps):
                error_temp.append(np.loadtxt(str(my_files[i])))
            error_DON['d={}'.format(latent_dim[k])] = np.array(error_temp)
        error_DON['Full\n DON'] = np.array([0.0005092, 0.0005122, 0.000481, 0.0004991, 0.000520]) # Benchmark DeepONet!!
        errors_DON_dict[method[j]] = error_DON
        
    if ood == 1:
        ### DeepONet OOD
        errors_OOD_dict = {}
        for j in range(len(method)):
            error_OOD = {}
            for k in range(len(latent_dim)):
                my_files = glob.glob('results/d_' + str(latent_dim[k]) + '/' + str(method[j]) + '/errors_OOD/' + '*.txt')
                error_temp = []
                for i in range(reps):
                    error_temp.append(np.loadtxt(str(my_files[i])))
                error_OOD['d={}'.format(latent_dim[k])] = np.array(error_temp)
            error_OOD['Full\n DON'] = np.array([0.0439, 0.0441, 0.042, 0.0433, 0.045]) # Benchmark DeepONet!!
            errors_OOD_dict[method[j]] = error_OOD
    
    if noise == 1:
        ### DeepONet Noise
        errors_NOISY_dict = {}
        for j in range(len(method)):
            error_NOISY = {}
            for k in range(len(latent_dim)):
                my_files = glob.glob('results/d_' + str(latent_dim[k]) + '/' + str(method[j]) + '/errors_NOISY/' + '*.txt')
                error_temp = []
                for i in range(reps):
                    error_temp.append(np.loadtxt(str(my_files[i])))
                error_NOISY['d={}'.format(latent_dim[k])] = np.array(error_temp)
            error_NOISY['Full\n DON'] = np.array([0.0494, 0.050, 0.0490, 0.0491, 0.0451]) # Benchmark DeepONet!!
            errors_NOISY_dict[method[j]] = error_NOISY
        
    # Create dataframes for each method
    pd_df_AE, pd_df_DON, pd_df_OOD, pd_df_NOISY = [], [], [], []
    for i in range(len(method)):
        pd_df_AE.append(pd.DataFrame(errors_AE_dict[method[i]]))
        pd_df_DON.append(pd.DataFrame(errors_DON_dict[method[i]]))
        if ood == 1:
            pd_df_OOD.append(pd.DataFrame(errors_OOD_dict[method[i]]))
        if noise == 1:
            pd_df_NOISY.append(pd.DataFrame(errors_NOISY_dict[method[i]]))
        
    np.savez(dir_final_res + 'final_errors_d_{}_method_{}.npz'.format(str(latent_dim), str(method)), ae=pd_df_AE, don=pd_df_DON, ood=pd_df_OOD, noisy=pd_df_NOISY) # Save errors to npz file
        
    return pd_df_AE, pd_df_DON, pd_df_OOD, pd_df_NOISY, errors_AE_dict, errors_DON_dict

# Create violin plots
def violin_plots(method, latent_dim, pd_df_AE, pd_df_DON, pd_df_OOD, pd_df_NOISY, save):
    '''
    Generates violin plots for AE & DON errors of all latent dimensions
    method: list of all methods
    pd_df_AE: pd dataframe with AE errors
    pd_df_DON: pd dataframe with DON errors
    '''
    # First plot - AE and DON
    for i in range(len(method)):

        sns.set(style = 'whitegrid') 

        plt.figure(figsize=(10,3.4))
        plt.subplot(1,2,1)
        ax = sns.violinplot(data=pd_df_AE[i], linewidths=0.5, linecolor='k')
        plt.title('Method: ' + method[i])
        plt.ylabel('MSE')
        #plt.yscale('log')
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

        plt.subplot(1,2,2)
        ax = sns.violinplot(data=pd_df_DON[i], linewidths=0.5, linecolor='k')
        plt.title('Latent DeepONet')
        #plt.yscale('log')
        plt.grid(which='minor', alpha=1)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        #ax.axvline(4.5, color='k', linestyle='--', lw=1)

        plt.tight_layout()
        
        if save==True:
            plt.savefig(dir_final_res + 'violin_plot_AE_DON_{}_d_{}.png'.format(method[i],latent_dim), bbox_inches='tight', dpi=500)
        
    if pd_df_OOD != [] and pd_df_NOISY != []:
        # Second plot - OOD and Noise
        for i in range(len(method)):

            sns.set(style = 'whitegrid') 

            plt.figure(figsize=(10,3.4))
            plt.subplot(1,2,1)
            ax = sns.violinplot(data=pd_df_OOD[i], linewidths=0.5, linecolor='k')
            plt.title('OOD data')
            plt.ylabel('MSE')
            #plt.yscale('log')
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            #ax.axvline(4.5, color='k', linestyle='--', lw=1)
        
            plt.subplot(1,2,2)
            ax = sns.violinplot(data=pd_df_NOISY[i], linewidths=0.5, linecolor='k')
            plt.title('Noisy data')
            #plt.yscale('log')
            plt.grid(which='minor', alpha=1)
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            #ax.axvline(4.5, color='k', linestyle='--', lw=1)

            plt.tight_layout()
        
            if save==True:
                plt.savefig(dir_final_res + 'violin_plot_OOD_Noise_{}_d_{}.png'.format(method[i],latent_dim), bbox_inches='tight', dpi=500)
        
        
        
