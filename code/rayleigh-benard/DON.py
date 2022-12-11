# Author: Katiana Kontolati
# Last update: Sept. 12, 2022
# Important: Make sure to: $conda activate latent_deeponet before running this script!!

import random
import tensorflow as tf
import keras
from keras import backend as K
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, add
from keras.layers import Layer, Dense, Dropout, Activation, Flatten, Reshape, BatchNormalization
from keras import regularizers
from keras.regularizers import l2
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from keras.layers.core import Lambda
from keras import layers
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
import numpy as np
from matplotlib import pylab as plt
from sklearn.metrics import mean_squared_error
import argparse
import os, sys, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # supress warnings

#### Parser
parser = argparse.ArgumentParser(description='Running autoencoder models.')
parser.add_argument(
    '--method',
    default='vanilla-AE',
    help='vanilla-AE | MLAE | CAE | WAE')
parser.add_argument(
    '--latent_dim',
    type=int,
    default=64,
    help='latent dimensionality (default: 64)')
parser.add_argument(
    '--n_samples',
    type=int,
    default=800,
    help='number of generated samples (default: 800)')
parser.add_argument(
    '--n_epochs',
    type=int,
    default=800,
    help='number of epochs (default: 800)')
parser.add_argument(
    '--bs',
    type=int,
    default=128,
    help='batch size (default: 128)')
parser.add_argument(
    '--ood',
    type=int,
    default=0,
    help='generate results for OOD data')
parser.add_argument(
    '--noise',
    type=int,
    default=0,
    help='generate results for noisy data')

args, unknown = parser.parse_known_args()

#### Fix random see (for reproducibility of results)
seed_value = random.randint(1,1000)
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
#tf.random.set_seed(seed_value)
#session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
#sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
#tf.compat.v1.keras.backend.set_session(sess)

# Load all data (original + reduced)
data_dir = 'results/d_' + str(args.latent_dim) + '/' + str(args.method) + '/data/'
file = np.load(data_dir + 'data_d_{}.npz'.format(args.latent_dim))

nt, nx, ny = 40, 128, 128
ld = int(file['latent_dim']) # latent dimension
ld_sqrt = int(np.sqrt(ld)) # square of latent dimension (for branch net convolutions)

# Load data (original + reduced inputs (x) and outputs (y))
x_train_red = file['x_train_red']
x_test_red = file['x_test_red']

y_train_red = file['y_train_red']
y_test_red = file['y_test_red']

x_train = file['x_train']
x_test = file['x_test']

y_train_og = file['y_train'] 
y_test_og = file['y_test']

num_train = x_train.shape[0]  # number of data
num_test = x_test.shape[0]

if args.ood == 1:
    # OOD and noisy data
    x_ood_red = file['x_ood_red']
    y_ood_red = file['y_ood_red']
    y_ood_og = file['y_ood'].reshape(x_ood_red.shape[0], nt*nx*ny)
    num_ood = x_ood_red.shape[0]
if args.noise == 1:
    x_noisy_red = file['x_noisy_red']
    y_noisy_red = file['y_noisy_red']
    y_noisy_og = file['y_noisy']
    num_noisy = x_noisy_red.shape[0]

if ld != args.latent_dim:
    print('Warning! The latent dimension of the saved file and the one given on prompt do not match!')

if not os.path.exists('results/d_' + str(args.latent_dim) + '/' + str(args.method) + '/errors_DON/'):
    os.makedirs('results/d_' + str(args.latent_dim) + '/' + str(args.method) + '/errors_DON/')     
if not os.path.exists('results/d_' + str(args.latent_dim) + '/' + str(args.method) + '/class_DON/'):
    os.makedirs('results/d_' + str(args.latent_dim) + '/' + str(args.method) + '/class_DON/') 
if args.ood == 1:    
    if not os.path.exists('results/d_' + str(args.latent_dim) + '/' + str(args.method) + '/errors_OOD/'):
        os.makedirs('results/d_' + str(args.latent_dim) + '/' + str(args.method) + '/errors_OOD/')   
if args.noise == 1:
    if not os.path.exists('results/d_' + str(args.latent_dim) + '/' + str(args.method) + '/errors_NOISY/'):
        os.makedirs('results/d_' + str(args.latent_dim) + '/' + str(args.method) + '/errors_NOISY/')       
    
# Load autoencoder class
class_AE_dir = 'results/d_' + str(args.latent_dim) + '/' + str(args.method) + '/class_AE/'

if args.method == 'vanilla-AE' or args.method == 'MLAE' or args.method == 'CAE':
    autoencoder = keras.models.load_model(class_AE_dir + 'class_AE_{}'.format(args.latent_dim))  # load autoencoder   
else:
    class WAE_Decoder(nn.Module):
        def __init__(self, args2):
            super(WAE_Decoder, self).__init__()

            self.n_channel = args2['n_channel']
            self.dim_h = args2['dim_h']
            self.n_z = args2['n_z']

            # first layer is fully connected
            self.fc = nn.Sequential(
                nn.Linear(self.n_z, self.dim_h * 8 * 7 * 7),  
                nn.ReLU()
            )

            # deconvolutional filters, essentially the inverse of convolutional filters
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(self.dim_h * 8, self.dim_h * 4, 4),
                nn.BatchNorm2d(self.dim_h * 4),
                nn.ReLU(True),
                nn.ConvTranspose2d(self.dim_h * 4, self.dim_h * 2, 4),
                nn.BatchNorm2d(self.dim_h * 2),
                nn.ReLU(True),
                nn.ConvTranspose2d(self.dim_h * 2, 1, 4, stride=2),
                nn.Sigmoid()
            )

        def forward(self, x):
            x = self.fc(x)
            x = x.view(-1, self.dim_h * 8, 7, 7)
            x = self.deconv(x)
            return x
    
    autoencoder = torch.load(class_AE_dir + 'final_decoder.pth') # load decoder
    
### DeepONet classes
class DeepONet_Model(tf.keras.Model):

    def __init__(self, Par):
        super(DeepONet_Model, self).__init__()

        #Defining some model parameters
        self.p = 4  # this is p 
        self.m = args.latent_dim # e.g., 64

        self.Par = Par

        self.index_list = []
        self.train_loss_list = []
        self.val_loss_list = []

        self.lr=10**-4
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

        self.branch_net_ls = self.build_branch_net()
        self.trunk_net_ls  = self.build_trunk_net()

        self.alpha = tf.Variable(1, trainable=True)

    def build_branch_net(self):
        ls=[]

        ls.append( Conv2D(32, (3,3), name='conv1', input_shape=[ld_sqrt, ld_sqrt, self.Par['n_channels']]) ) 
        ls.append( Activation(tf.math.sin)  )
        ls.append( BatchNormalization() )

        if self.m > 16:  # reduce conv layers for small latent dimensions
            ls.append( Conv2D(16, (3,3), name='conv2') ) #[10,10,16]
            ls.append( Activation(tf.math.sin)  )
            ls.append( BatchNormalization() )
        
        if self.m > 36:  # reduce conv layers for small latent dimensions
            ls.append( Conv2D(16, (3,3), name='conv3') ) #[8,8,16]
            ls.append( Activation(tf.math.sin)  )
            ls.append( BatchNormalization() )
    
        if self.m > 64:  # reduce conv layers for small latent dimensions
            ls.append( Conv2D(16, (3,3), name='conv4') ) #[8,8,16]
            ls.append( Activation(tf.math.sin)  )
            ls.append( BatchNormalization() )

        ls.append( Flatten() )
        ls.append(Dense(self.m*self.p))

        return ls

    def build_trunk_net(self):
        ls=[]

        ls.append( Dense(100))
        ls.append( Activation(tf.math.sin)  )

        ls.append( Dense(100))
        ls.append( Activation(tf.math.sin)  )

        ls.append(Dense(self.m*self.p))

        return ls

    @tf.function(jit_compile=True)
    def call(self, X_func, X_loc):
    #X_func -> [BS*n_t, k*n_f]
    #X_loc  -> [n_t, 1]

        n_t = X_loc.shape[0]

        y_func = X_func
        y_func = (y_func - self.Par['mean'])/self.Par['std']

        for i in range(len(self.branch_net_ls)):
            y_func = self.branch_net_ls[i](y_func)
        
        y_loc = 10*(X_loc-0.5)
        for i in range(len(self.trunk_net_ls)):
            y_loc = self.trunk_net_ls[i](y_loc)

        y_func = tf.reshape(y_func, [-1, self.m, self.p])
        y_loc = tf.reshape(y_loc, [-1, self.m, self.p])

        Y = tf.einsum('ijk,pjk->ipj', y_func, y_loc)

        return(Y)

    @tf.function(jit_compile=True)
    def loss(self, y_pred, y_train):

        # mse = MeanSquaredError()
        #Total Loss
        train_loss =  tf.reduce_mean( tf.square( y_pred - y_train ) )
        #mse(y_pred, y_train)

        return([train_loss])


def preprocess(x, y):
    # Gives one datapoint back 
    n = x.shape[0]
    X_func = x.reshape(n, ld_sqrt, ld_sqrt, 1) # (#, sqrt(ld), sqrt(ld), 1)
    #print(X_func.shape)

    X_loc = np.linspace(0, 1, nt).reshape(nt, 1)
    #print(X_loc.shape)

    y = y.reshape(-1, nt, args.latent_dim)
    #print(y.shape)

    return X_func, X_loc, y

def tensor(x):
    return tf.convert_to_tensor(x, dtype=tf.float32)


@tf.function(jit_compile=True)
def train(don_model, X_func, X_loc, y):
    with tf.GradientTape() as tape:
        y_hat  = don_model(X_func, X_loc)
        loss   = don_model.loss(y_hat, y)[0]

    gradients = tape.gradient(loss, don_model.trainable_variables)
    don_model.optimizer.apply_gradients(zip(gradients, don_model.trainable_variables))
    return(loss)

def error_metric(true, pred):
    #true - [samples, time steps, 128, 128]
    #pred - [samples, time steps, 128, 128]
    pred = np.reshape(pred, (-1, nt*nx*ny))
    true = np.reshape(true, (-1, nt*nx*ny))

    mse = mean_squared_error(true, pred)

    return mse

def show_error(don_model, ae_model, X_func, X_loc, pf_true, n_samples, save, class_dir):
    y_pred = don_model(X_func, X_loc)
    y_pred = np.reshape(y_pred, (-1, args.latent_dim))
    
    if args.method == 'WAE':
        y_pred = torch.Tensor(y_pred)
        pf_pred = autoencoder.forward(y_pred).detach().numpy().reshape(n_samples, nt, nx, ny)
    else:
        pf_pred = autoencoder.decoder(y_pred).numpy()
        pf_pred = pf_pred.reshape(n_samples, nt, nx, ny)    
    
    if save==True:
        np.savez(class_dir + 'results_test_data.npz', ref=pf_true, pred=pf_pred)
    
    error = error_metric(pf_true, pf_pred)
    print('Mean squared error (MSE): ', error)
    
    return error

def main():
    
    Par = {}
    class_DON_dir = 'results/d_' + str(args.latent_dim) + '/' + str(args.method) + '/class_DON/'
    Par['address'] = class_DON_dir

    #print(Par['address'])
    #print('------\n')

    X_func_train, X_loc_train, y_train = preprocess(x_train_red, y_train_red)
    X_func_test, X_loc_test, y_test = preprocess(x_test_red, y_test_red)
    if args.ood == 1:
        X_func_ood, X_loc_ood, y_ood = preprocess(x_ood_red, y_ood_red)
    if args.noise == 1:
        X_func_noisy, X_loc_noisy, y_noisy = preprocess(x_noisy_red, y_noisy_red)
    
    Par['n_channels'] = X_func_train.shape[-1]
    #print(Par['n_channels'] )

    #print('X_func_train: ', X_func_train.shape, '\nX_loc_train: ', X_loc_train.shape, '\ny_train: ', y_train.shape)
    #print('X_func_test: ', X_func_test.shape, '\nX_loc_test: ', X_loc_test.shape, '\ny_test: ', y_test.shape)
    
    Par['mean'] = np.mean(X_func_train)
    Par['std'] =  np.std(X_func_train)

    #print('mean: ', Par['mean'])
    #print('std : ', Par['std'])
    
    don_model = DeepONet_Model(Par)
    n_epochs = args.n_epochs
    batch_size = args.bs
    
    print('DeepONet training in progress...')
    
    begin_time = time.time()
    
    for i in range(n_epochs+1):
        
        for end in np.arange(batch_size, X_func_train.shape[0]+1, batch_size):  
            
            start = end - batch_size
            loss = train(don_model, tensor(X_func_train[start:end]), tensor(X_loc_train), tensor(y_train[start:end]))
            
        if i%1 == 0:

            #don_model.save_weights(Par['address'] + "/model_"+str(i))
            train_loss = loss.numpy()
            y_hat = don_model(X_func_test, X_loc_test)
            val_loss = np.mean( (y_hat - y_test)**2 )

            #print("epoch:" + str(i) + ", Train Loss:" + "{:.3e}".format(train_loss) + 
            #      ", Val Loss:" + "{:.3e}".format(val_loss) +  ", elapsed time: " 
            #       +  str(int(time.time()-begin_time)) + "s"  )

            don_model.index_list.append(i)
            don_model.train_loss_list.append(train_loss)
            don_model.val_loss_list.append(val_loss)

        if i == n_epochs:
            don_model.save_weights(Par['address'] + "/model_"+str(i)) # save last model

    # Compute relative L2 error between reference and prediction
    pred = tf.reshape(y_hat, [num_test, nt*args.latent_dim])
    ref = tf.reshape(y_test, [num_test, nt*args.latent_dim])
    errors = np.abs(pred - ref)
    l2_rel_err = np.linalg.norm(errors, axis=1)/np.linalg.norm(ref, axis=1)
    l2 = np.mean(l2_rel_err)
    print('Latent DeepONet relative L2 error on test data: {}\n'.format(round(l2,4)))
            
    #Convergence plot
    index_list = don_model.index_list
    train_loss_list = don_model.train_loss_list
    val_loss_list = don_model.val_loss_list
    np.savez(Par['address']+'/convergence_data', index_list=index_list, 
                 train_loss_list=train_loss_list, val_loss_list=val_loss_list)

    plt.close()
    plt.figure(figsize=(4,3))
    plt.plot(index_list, train_loss_list, label="train", linewidth=2)
    plt.plot(index_list, val_loss_list, label="validation", linewidth=2)
    plt.legend(['train', 'validation'], loc='upper right')
    plt.yscale('log')
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.title('Latent DeepONet loss')
    plt.tight_layout()
    plt.savefig(Par["address"] + "/convergence.png", dpi=800)
    plt.close()

    if True:
        
        # Load best DeepONet model
        don_model = DeepONet_Model(Par)
        #don_model_number = index_list[np.argmin(val_loss_list)]
        don_model_number = n_epochs
        np.save(Par['address'] + 'best_don_model_number', don_model_number)
        don_model_address = Par['address'] + "/model_"+str(don_model_number)
        don_model.load_weights(don_model_address)

        print('best DeepONet model: ', don_model_number)

        n_samples = 20 # number of samples to compare

        # Train
        pf_true = y_train_og       # load original data 
        pf_true = (pf_true - np.min(pf_true))/(np.max(pf_true) - np.min(pf_true))
        pf_true = np.reshape(pf_true, (-1,nt,nx,ny))
        pf_true_train = pf_true[:n_samples, :]

        # Test
        pf_true = y_test_og        
        pf_true = (pf_true - np.min(pf_true))/(np.max(pf_true) - np.min(pf_true))
        pf_true = np.reshape(pf_true, (-1,nt,nx,ny))
        pf_true_test = pf_true[:n_samples, :]
        
        if args.ood == 1:
            # OOD
            pf_true = y_ood_og        
            pf_true = (pf_true - np.min(pf_true))/(np.max(pf_true) - np.min(pf_true))
            pf_true = np.reshape(pf_true, (-1,nt,nx,ny))
            pf_true_ood = pf_true[:num_ood, :]
        
        if args.noise == 1:
            # Noisy
            pf_true = y_noisy_og        
            pf_true = (pf_true - np.min(pf_true))/(np.max(pf_true) - np.min(pf_true))
            pf_true = np.reshape(pf_true, (-1,nt,nx,ny))
            pf_true_noisy = pf_true[:num_noisy, :]
        
        X_loc = np.linspace(0,1,nt)[:,None]
        
        print('')
        print('Train Dataset')
        error_train = show_error(don_model, autoencoder, X_func_train[:n_samples], X_loc, pf_true_train, n_samples, save=False, class_dir=class_DON_dir)

        print('Test Dataset')
        error_test = show_error(don_model, autoencoder, X_func_test[:n_samples], X_loc, pf_true_test, n_samples, save=True, class_dir=class_DON_dir)

        if args.ood == 1:
            print('OOD Dataset')
            error_ood = show_error(don_model, autoencoder, X_func_ood[:num_ood], X_loc, pf_true_ood, num_ood, save=False, class_dir=class_DON_dir)
        
        if args.noise == 1:
            print('Noisy Dataset')
            error_noisy = show_error(don_model, autoencoder, X_func_noisy[:num_noisy], X_loc, pf_true_noisy, num_noisy, save=False, class_dir=class_DON_dir)
        
        # Save test error
        errordir = 'results/d_' + str(args.latent_dim) + '/' + str(args.method) + '/errors_DON/'
        np.savetxt(errordir + 'error_DON_seed_' + str(seed_value) + '.txt', 
                                   np.expand_dims(np.array(error_test), axis=0), fmt='%e') 
        if args.ood == 1:
            error_ood_dir = 'results/d_' + str(args.latent_dim) + '/' + str(args.method) + '/errors_OOD/'
            np.savetxt(error_ood_dir + 'error_DON_seed_' + str(seed_value) + '.txt', 
                                   np.expand_dims(np.array(error_ood), axis=0), fmt='%e') 
        if args.noise == 1:
            error_noisy_dir = 'results/d_' + str(args.latent_dim) + '/' + str(args.method) + '/errors_NOISY/'                
            np.savetxt(error_noisy_dir + 'error_DON_seed_' + str(seed_value) + '.txt', 
                                   np.expand_dims(np.array(error_noisy), axis=0), fmt='%e') 
        
        print('-------------------Complete-------------------\n')
        
# Run main
main()




