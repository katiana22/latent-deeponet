import numpy as np
from matplotlib import pylab as plt
import argparse
import os, sys, glob
import pandas as pd
import seaborn as sns

if not os.path.exists('results/final_results/'):
    os.makedirs('results/final_results/') 

dir_final_res = 'results/final_results/'  
    
def errors_ae_don(latent_dim, method, reps):
    
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
        error_DON['Full\n DON'] = np.array([0.0299, 0.0281, 0.0314, 0.0311, 0.0290]) # Benchmark DeepONet!!
        errors_DON_dict[method[j]] = error_DON
        
    # Create dataframes for each method
    pd_df_AE = []
    pd_df_DON = []
    for i in range(len(method)):
        pd_df_AE.append(pd.DataFrame(errors_AE_dict[method[i]]))
        pd_df_DON.append(pd.DataFrame(errors_DON_dict[method[i]]))
        
    np.savez(dir_final_res + 'final_errors.npz', ae=pd_df_AE, don=pd_df_DON) # Save errors to npz file
        
    return pd_df_AE, pd_df_DON, errors_AE_dict, errors_DON_dict

# Create violin plots
def violin_plots(method, pd_df_AE, pd_df_DON, save):
    '''
    Generates violin plots for AE & DON errors of all latent dimensions
    method: list of all methods
    pd_df_AE: pd dataframe with AE errors
    pd_df_DON: pd dataframe with DON errors
    '''
    for i in range(len(method)):

        sns.set(style = 'whitegrid') 

        plt.figure(figsize=(8,3.5))
        plt.subplot(1,2,1)
        ax = sns.violinplot(data=pd_df_AE[i], linewidths=0.5, linecolor='gray')
        plt.title('Method: ' + method[i])
        plt.ylabel('Relative L2 error')
        #plt.yscale('log')
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

        plt.subplot(1,2,2)
        ax = sns.violinplot(data=pd_df_DON[i], linewidths=0.5, linecolor='gray')
        plt.title('Latent DeepONet')
        #plt.yscale('log')
        plt.grid(which='minor', alpha=1)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        #ax.axvline(2.6, color='k', linestyle='--', lw=1)

        plt.tight_layout()
        
        if save==True:
            plt.savefig(dir_final_res + 'violin_plot_{}.png'.format(method[i]), bbox_inches='tight', dpi=500)
        
        
