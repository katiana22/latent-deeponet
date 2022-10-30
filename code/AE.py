# Author: Katiana Kontolati
# Last update: Sept. 12, 2022
# Important: Make sure to: $conda activate latent_deeponet before running this script!!

import keras
import numpy as np
import matplotlib.pyplot as plt
#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')
import tensorflow as tf
#tf.get_logger().setLevel('WARNING')
import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import random
from keras import backend as K
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, add
from keras.layers import Layer, Dense, Dropout, Activation, Flatten, Reshape
from keras import regularizers
from keras.regularizers import l2
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from keras.layers.core import Lambda
from keras import layers
#from keras.objectives import binary_crossentropy
import argparse
#from IPython.display import display
#from IPython.display import clear_output
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
#from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import seaborn as sns
import subprocess
import PIL
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import argparse
import os, sys
import scipy.io
from sklearn.metrics import mean_squared_error
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # supress warnings

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
#K.set_session(sess)

#### Load data
nx, ny, nt = 256, 256, 72
n = 271 # all data

file = np.load('../../scr16_mshiel10/kontolati/shallow/data/shallow-water-1.npz')
inputs_1, outputs_1 = file['inputs'], np.array((file['outputs']))

file = np.load('../../scr16_mshiel10/kontolati/shallow/data/shallow-water-2.npz')
inputs_2, outputs_2 = file['inputs'], np.array((file['outputs']))

inputs = np.concatenate((inputs_1, inputs_2), axis=0).reshape(n, nx*ny) # flatten
outputs = np.concatenate((outputs_1, outputs_2), axis=0)

outputs = outputs[:,:nt,:,:]  # keep less time steps

outputs_re = outputs.reshape(n*nt, nx*ny) # reshape

n_samples = inputs.shape[0]
num_train = 240
num_test = n - num_train
outputs_re = outputs.reshape(n*nt, nx*ny) # reshape
x_y_data = np.concatenate((inputs, outputs_re), axis=0) # concatenate inputs and outputs

if args.ood == 1:
    # OOD data
    file2 = np.load('./Data/Brusselator_data_KLE_lx_0.45_ly_0.4_v_0.15.npz') # OOD
    n_samples_ood, num_ood = file2['n_samples'], 50
    inputs_ood, outputs_ood = file2['inputs'].reshape(n_samples_ood, nx*ny)[:num_ood,:], np.array((file2['outputs'])).reshape(n_samples_ood, nt, nx, ny)[:num_ood,:,:,:]
    outputs_ood_re = outputs_ood.reshape(num_ood*nt, nx*ny)
    x_y_ood_data = np.concatenate((inputs_ood, outputs_ood_re), axis=0) # concatenate inputs and outputs

if args.noise == 1:
    # Make noisy inputs (add 10% uncorrelated noise to inputs)
    num_noisy, noise = 50, 0.1
    inputs_n, outputs_n = file['inputs'][n:n+num_noisy,:], np.array((file['outputs']))[n:n+num_noisy,:]
    inputs_n = inputs_n.reshape(num_noisy, nx, ny)
    inputs_mean = np.mean(inputs_n)
    outputs_n_re = outputs_n.reshape(num_noisy*nt, nx*ny)

    x_noisy = inputs_n + np.random.normal(0, noise*inputs_mean, (num_noisy, nx, ny))
    x_noisy = x_noisy.reshape(num_noisy, nx*ny)
    x_y_noisy_data = np.concatenate((x_noisy, outputs_n_re), axis=0)  # concatenate

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# Normalize datasets
x_y_data_norm = NormalizeData(x_y_data).astype("float32") 
if args.ood == 1:
    x_y_ood_data_norm = NormalizeData(x_y_ood_data).astype("float32") 
if args.noise == 1:
    x_y_noisy_data_norm = NormalizeData(x_y_noisy_data).astype("float32") 
    x_noisy_data_norm = x_y_noisy_data_norm[:num_noisy,:]
    y_noisy_data_norm = x_y_noisy_data_norm[num_noisy:,:]

x_norm = x_y_data_norm[:n]
y_norm = x_y_data_norm[n:]

#print('All data shape:', x_norm.shape, y_norm.shape)

# Split to train/eval autoencoder
x_y_train, x_y_test = train_test_split(x_y_data_norm, test_size=0.1, random_state=42)
n_train, n_test, n_all = x_y_train.shape[0], x_y_test.shape[0], x_y_data.shape[0] # for autoencoder training

x_y_train = x_y_train.reshape(n_train, nx, ny) # reshape
x_y_test = x_y_test.reshape(n_test, nx, ny)

# print('')
# print('All data train (reshaped):', x_y_train.shape)
# print('All data test (reshaped):', x_y_test.shape)


#### Prepare data 
if args.method == 'CAE':
    x_y_train = x_y_train.reshape(n_train, nx, ny, 1) # (#, 28, 28, 1) important for convolutional layers!
    x_y_test = x_y_test.reshape(n_test, nx, ny, 1)
    
elif args.method == 'WAE':
    
    x_y_train = x_y_train.reshape(n_train, nx, ny) # (#, 28, 28) important!
    x_y_test = x_y_test.reshape(n_test, nx, ny)
    if args.ood == 1:
        x_y_ood = x_y_ood_data_norm.reshape(num_ood*nt+num_ood, nx, ny)
    if args.noise == 1:
        x_noisy_data_norm = x_noisy_data_norm.reshape(num_noisy, nx, ny)
        y_noisy_data_norm = y_noisy_data_norm.reshape(num_noisy*nt, nx, ny)
    
    x_y_train = np.expand_dims(x_y_train, axis=1) # Reshape (bs, 1, 28, 28)
    x_y_test = np.expand_dims(x_y_test, axis=1)
    if args.ood == 1:
        x_y_ood = np.expand_dims(x_y_ood, axis=1)
    if args.noise == 1:
        x_noisy_data_norm = np.expand_dims(x_noisy_data_norm, axis=1)
        y_noisy_data_norm = np.expand_dims(y_noisy_data_norm, axis=1)
    x_y_all = np.concatenate((x_y_train, x_y_test))
    
    args2 = {}
    args2['n_z'] = args.latent_dim   # latent dim!
    args2['dim_h'] = 5         # factor controlling size of hidden layers
    args2['n_channel'] = 1      # number of channels in the input data (MNIST = 1, greyscale)
    args2['sigma'] = 1.0        # variance in n_z
    args2['lambda'] = 0.01      # hyper param for weight of discriminator loss
    args2['lr'] = 0.002        # learning rate for Adam optimizer
    args2['epochs'] = args.n_epochs  # how many epochs to run for
    args2['batch_size'] = args.bs # batch size for SGD
    args2['save'] = True       # save weights at each epoch of training if TrueUniform(loc=true_mean2, scale=true_std2) 
    args2['train'] = True      # train networks if True, else load networks from saved weights
    args2['dataset'] = 'custom'  # specify which dataset to use
    
    # Transform data to torch datasets
    tensor_train = torch.Tensor(x_y_train) # transform to torch tensor
    tensor_test = torch.Tensor(x_y_test)
    if args.ood == 1:
        tensor_ood = torch.Tensor(x_y_ood)
    if args.noise == 1:
        tensor_x_noisy = torch.Tensor(x_noisy_data_norm)
        tensor_y_noisy = torch.Tensor(y_noisy_data_norm)
    tensor_all = torch.Tensor(x_y_all)

    trainset = TensorDataset(tensor_train) # create your datset
    testset = TensorDataset(tensor_test)
    allset = TensorDataset(tensor_all)

    train_loader = DataLoader(
        dataset=trainset,
        batch_size=args2['batch_size'],
        shuffle=False)

    test_loader = DataLoader(
        dataset=testset,
        batch_size=args2['batch_size'],
        shuffle=False)

    test_all = DataLoader(
        dataset=allset,
        batch_size=args2['batch_size'],
        shuffle=False)
    
else:
    x_y_train = x_y_train.reshape(n_train, nx, ny) # (#, 28, 28) important!
    x_y_test = x_y_test.reshape(n_test, nx, ny)
    

#### Create directories for results
if not os.path.exists('results/d_' + str(args.latent_dim) + '/' + str(args.method) + '/plots/'):
    os.makedirs('results/d_' + str(args.latent_dim) + '/' + str(args.method) + '/plots/')
if not os.path.exists('results/d_' + str(args.latent_dim) + '/' + str(args.method) + '/errors_AE/'):
    os.makedirs('results/d_' + str(args.latent_dim) + '/' + str(args.method) + '/errors_AE/')
if not os.path.exists('results/d_' + str(args.latent_dim) + '/' + str(args.method) + '/data/'):
    os.makedirs('results/d_' + str(args.latent_dim) + '/' + str(args.method) + '/data/')
if not os.path.exists('results/d_' + str(args.latent_dim) + '/' + str(args.method) + '/class_AE/'):
    os.makedirs('results/d_' + str(args.latent_dim) + '/' + str(args.method) + '/class_AE/')    
    
class_AE_dir = 'results/d_' + str(args.latent_dim) + '/' + str(args.method) + '/class_AE/'

#### Run autoencoders
print('Seed number:', seed_value)
print('Autoencoder training in progress...')
    
### Classes

class vanilla_Autoencoder(Model):
    def __init__(self, latent_dim):
        super(vanilla_Autoencoder, self).__init__()
        self.latent_dim = latent_dim   
        self.encoder = tf.keras.Sequential([
          layers.Flatten(),
          layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
          layers.Dense(nx*ny, activation='sigmoid'),
          layers.Reshape((nx, ny))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
class MLAE(Model):
    def __init__(self, latent_dim):
        super(MLAE, self).__init__()
        self.latent_dim = latent_dim   
        self.encoder = tf.keras.Sequential([
          layers.Flatten(),
          layers.Dense(256, activation='relu'),
          layers.Dense(169, activation='relu'),
          layers.Dense(121, activation='relu'),
          layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
          layers.Dense(121, activation='relu'),
          layers.Dense(169, activation='relu'),
          layers.Dense(256, activation='relu'),
          layers.Dense(nx*ny, activation='sigmoid'), 
          layers.Reshape((nx, ny))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
class CAE(Model):
    def __init__(self):
        super(CAE, self).__init__()
        self.encoder = tf.keras.Sequential([
          layers.Input(shape=(nx, ny, 1)),
          layers.Conv2D(8, (3, 3), activation="relu", padding="same"),
          layers.MaxPooling2D((2, 2), padding="same"),
          layers.Conv2D(8, (3, 3), activation="relu", padding="same"),
          layers.MaxPooling2D((2, 2), padding="same"),
          layers.Conv2D(4, (3, 3), activation="relu", padding="same"),
          layers.MaxPooling2D((2, 2), padding="same"),
          layers.Conv2D(2, (3, 3), activation="relu", padding="same"),
          layers.MaxPooling2D((2, 2), padding="same"),
          layers.Flatten(),
          layers.Dense(args.latent_dim, activation="linear")])
             
        self.decoder = tf.keras.Sequential([
          layers.Dense(16*16*4, activation="relu"),
          layers.Reshape(target_shape=(16,16,4)),
          layers.Conv2DTranspose(2, kernel_size=3, strides=2, activation="relu", padding="same"),
          layers.Conv2DTranspose(4, kernel_size=3, strides=2, activation="relu", padding="same"),
          layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation="relu", padding="same"),
          layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation="relu", padding="same"),
          layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded   
    

#### WAE
class WAE_Encoder(nn.Module):
    def __init__(self, args2):
        super(WAE_Encoder, self).__init__()

        self.n_channel = args2['n_channel']
        self.dim_h = args2['dim_h']
        self.n_z = args2['n_z'] # latent dimension
        
        # convolutional filters, work excellent with image data
        self.conv = nn.Sequential(
            nn.Conv2d(self.n_channel, self.dim_h, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(self.dim_h, self.dim_h * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),
            nn.Conv2d(self.dim_h * 2, self.dim_h * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.ReLU(True),
            nn.Conv2d(self.dim_h * 4, self.dim_h * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 8),
            nn.ReLU(True),
        )
        
        # final layer is fully connected
        self.fc = nn.Linear(self.dim_h * (2 ** 3), self.n_z)

    def forward(self, x):
        x = self.conv(x)
        x = x.squeeze()
        x = self.fc(x)
        return x

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

# define the descriminator
class Discriminator(nn.Module):
    def __init__(self, args2):
        super(Discriminator, self).__init__()

        self.dim_h = args2['dim_h']
        self.n_z = args2['n_z']

        # main body of discriminator, returns [0,1]
        self.main = nn.Sequential(
            nn.Linear(self.n_z, self.dim_h * 4),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 4, self.dim_h * 4),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 4, self.dim_h * 4),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 4, self.dim_h * 4),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.main(x)
        return x
    
# control which parameters are frozen / free for optimization
def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True

def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False

if args.method == 'WAE':        
    def calc_test_loss(criterion=nn.MSELoss(), args2=args2,
                       model_name_list = ["WAE"],
                       encoder_class_list = [WAE_Encoder], 
                       decoder_class_list = [WAE_Decoder], 
                       encoder_filepath_list = [class_AE_dir+'save_weights/WAEgan_encoder-best_{}.pth'.format(args2['dataset'])],
                       decoder_filepath_list = [class_AE_dir+'save_weights/WAEgan_decoder-best_{}.pth'.format(args2['dataset'])]):

        ## load full test set as batch size calculating loss
        test_loader = DataLoader(dataset=allset, batch_size=n_all, shuffle=False)

        # Create dict to store test loss by model
        test_loss_dict = dict() 

        ## Calculate reconstruction test loss for each model
        for Encoder, Decoder, enc_filepath, dec_filepath, model in zip(encoder_class_list, 
                                                                       decoder_class_list, 
                                                                       encoder_filepath_list, 
                                                                       decoder_filepath_list, 
                                                                       model_name_list):

            # call encoder and decoder classes and eval each
            encoder, decoder = Encoder(args2), Decoder(args2)
            encoder.eval()
            decoder.eval()

            # load encoder and decoder weights from checkpoint
            enc_checkpoint = torch.load(enc_filepath)
            encoder.load_state_dict(enc_checkpoint)

            dec_checkpoint = torch.load(dec_filepath)
            decoder.load_state_dict(dec_checkpoint)

            # calculate and save test reconstruction loss for WAE

            plot_images = []
            z_hat_save = []
            for images in test_loader:
                images = images[0]
                z_hat = encoder(images)
                z_hat_save.append(z_hat) # save data in latent space

                x_hat = decoder(z_hat)
                plot_images.append(x_hat.detach().numpy())
                test_loss_dict[model] = criterion(x_hat, images).data.item()
                print('{0} final reconstruction loss {1} on {2}'.format(model, test_loss_dict[model], args2['dataset']))

        return plot_images, z_hat_save, test_loss_dict, encoder, decoder


#### Autoencoder

if args.method == 'vanilla-AE': 
    autoencoder = vanilla_Autoencoder(args.latent_dim)
    autoencoder.compile(optimizer='adam', loss='mse')
    
elif args.method == 'MLAE': 
    autoencoder = MLAE(args.latent_dim)
    autoencoder.compile(optimizer='adam', loss='mse') 

elif args.method == 'CAE':
    autoencoder = CAE()
    autoencoder.compile(optimizer='adam', loss='mse')
    

elif args.method == 'WAE':
    
    # Remove all previous weights
    command = ['rm', '-r', class_AE_dir + 'save_weights']
    subprocess.run(command)

    command = ['mkdir', class_AE_dir + 'save_weights']
    subprocess.run(command)

    ####################### WAE #######################
    # instantiate discriminator model, and restart encoder and decoder, for fairness. Set to train mode, etc
    wae_encoder, wae_decoder, discriminator = WAE_Encoder(args2), WAE_Decoder(args2), Discriminator(args2)

    criterion = nn.MSELoss()

    history_loss_train, history_loss_test = [], [] # save loss

    if args2['train']:
        enc_optim = torch.optim.Adam(wae_encoder.parameters(), lr = args2['lr'])
        dec_optim = torch.optim.Adam(wae_decoder.parameters(), lr = args2['lr'])
        dis_optim = torch.optim.Adam(discriminator.parameters(), lr = args2['lr'])

        enc_scheduler = torch.optim.lr_scheduler.StepLR(enc_optim, step_size=30, gamma=0.5)
        dec_scheduler = torch.optim.lr_scheduler.StepLR(dec_optim, step_size=30, gamma=0.5)
        dis_scheduler = torch.optim.lr_scheduler.StepLR(dis_optim, step_size=30, gamma=0.5)

        # one and -one allow us to control descending / ascending gradient descent
        one = torch.tensor(1, dtype=torch.float)

        for epoch in range(args2['epochs']):

            # train for one epoch -- set nets to train mode
            wae_encoder.train()
            wae_decoder.train()
            discriminator.train()

            # Included are elements similar to the Schelotto (2018) implementation
            # on GitHub. Schelotto's implementation repository is worth looking into, 
            # because the WAE-MMD ("Maximum Mean Discrepancy") implementation, a second 
            # WAE algorithm discussed in the original Wasserstein Auto-Encoders paper,
            # is also implemented there.

            for images in tqdm(train_loader, disable=True):
                images = images[0]

                # zero gradients for each batch
                wae_encoder.zero_grad()
                wae_decoder.zero_grad()
                discriminator.zero_grad()

                #### TRAIN DISCRIMINATOR ####

                # freeze auto encoder params
                frozen_params(wae_decoder)
                frozen_params(wae_encoder)

                # free discriminator params
                free_params(discriminator)

                # run discriminator against randn draws
                z = torch.randn(images.size()[0], args2['n_z']) * args2['sigma']
                d_z = discriminator(z)

                # run discriminator against encoder z's
                z_hat = wae_encoder(images)
                d_z_hat = discriminator(z_hat)

                d_z_loss = args2['lambda']*torch.log(d_z).mean()
                d_z_hat_loss = args2['lambda']*torch.log(1 - d_z_hat).mean()

                # formula for ascending the descriminator -- -one reverses the direction of the gradient.
                d_z_loss.backward(-one)
                d_z_hat_loss.backward(-one)

                dis_optim.step()

                #### TRAIN GENERATOR ####

                # flip which networks are frozen, which are not
                free_params(wae_decoder)
                free_params(wae_encoder)
                frozen_params(discriminator)

                batch_size = images.size()[0]

                # run images
                z_hat = wae_encoder(images)
                x_hat = wae_decoder(z_hat)

                # discriminate latents
                z_hat2 = wae_encoder(Variable(images.data))
                d_z_hat = discriminator(z_hat2)

                # calculate reconstruction loss
                # WAE is happy with whatever cost function, let's use BCE
                #kl_div
                #cross_entropy
                #mse_loss
                #binary_cross_entropy
                #binary_cross_entropy_with_logits
                BCE = nn.functional.binary_cross_entropy_with_logits(
                    x_hat.view(-1, nx*ny), 
                    images.view(-1, nx*ny), 
                    reduce=False
                ).mean()

                # calculate discriminator loss
                d_loss = args2['lambda'] * (torch.log(d_z_hat)).mean()

                # we keep the BCE and d_loss on separate graphs to increase efficiency in pytorch
                BCE.backward(one)
                # -one reverse the direction of the gradient, minimizing BCE - d_loss
                d_loss.backward(-one)

                enc_optim.step()
                dec_optim.step()

            # test on test set
            wae_encoder.eval()
            wae_decoder.eval()
            for images in tqdm(test_loader):
                images = images[0]

                z_hat = wae_encoder(images)
                x_hat = wae_decoder(z_hat)
                test_recon_loss = criterion(x_hat, images)


            if args2['save']:
                save_path = class_AE_dir + 'save_weights/WAEgan_{}-epoch_{}.pth'
                torch.save(wae_encoder.state_dict(), save_path.format('encoder', epoch))
                torch.save(wae_decoder.state_dict(), save_path.format('decoder', epoch))
                torch.save(discriminator.state_dict(), save_path.format('discriminator', epoch))

            # print stats after each epoch
            #print("Epoch: [{}/{}], \tTrain Reconstruction Loss: {} d loss: {}, \n"\
            #      "\t\t\tTest Reconstruction Loss:{}".format(
            #    epoch + 1, 
            #    args2['epochs'], 
            #    BCE.data.item(),
            #    d_loss.data.item(),
            #    test_recon_loss.data.item()
            #))

            history_loss_train.append(test_recon_loss.data.item())
            history_loss_test.append(BCE.data.item())

    else:
        enc_checkpoint = torch.load(class_AE_dir + 'save_weights/WAEgan_encoder-best_{}.pth'.format(args2['dataset']))
        wae_encoder.load_state_dict(enc_checkpoint)

        dec_checkpoint = torch.load(class_AE_dir + 'save_weights/WAEgan_decoder-best_{}.pth'.format(args2['dataset']))
        wae_decoder.load_state_dict(dec_checkpoint)

        dec_checkpoint = torch.load(class_AE_dir + 'save_weights/WAEgan_discriminator-best_{}.pth'.format(args2['dataset']))
        discriminator.load_state_dict(dec_checkpoint)

    # Rename best weights
    command = ['mv', class_AE_dir + 'save_weights/WAEgan_encoder-epoch_{}.pth'.format(args2['epochs']-1), class_AE_dir + 'save_weights/WAEgan_encoder-best_custom.pth']
    subprocess.run(command)

    command = ['mv', class_AE_dir + 'save_weights/WAEgan_decoder-epoch_{}.pth'.format(args2['epochs']-1), class_AE_dir + 'save_weights/WAEgan_decoder-best_custom.pth']
    subprocess.run(command)

    command = ['mv', class_AE_dir + 'save_weights/WAEgan_discriminator-epoch_{}.pth'.format(args2['epochs']-1), class_AE_dir + 'save_weights/WAEgan_discriminator-best_custom.pth']
    subprocess.run(command)
    
    plot_images, z_hat_tensor, test_loss, encoder, decoder = calc_test_loss()

if args.method == 'vanilla-AE' or args.method == 'MLAE' or args.method == 'CAE':

    # Train model
    epochs = args.n_epochs
    batch_size = args.bs

    history = autoencoder.fit(x_y_train, x_y_train, shuffle=True,
                            batch_size=batch_size, 
                            epochs=epochs, verbose=0, 
                            validation_data=(x_y_test, x_y_test))

    print('Autoencoder training is completed.')

    plt.figure(figsize=(4,3))
    plt.plot(history.history['loss'], linewidth=2)
    plt.plot(history.history['val_loss'], linewidth=2)
    plt.title('Autoencoder loss')
    plt.ylabel('MSE')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.yscale('log')
    plt.tight_layout()
    plotdir = 'results/d_' + str(args.latent_dim) + '/' + str(args.method) + '/plots/'
    plt.savefig(plotdir + 'History_' + str(args.method) + '_seed_' + str(seed_value) + '.png', dpi=300)

    # Compute L2 error on test data
    decoded_data = autoencoder.predict(x_y_test).reshape(n_test, nx*ny)
    reference_data = x_y_test.reshape(n_test, nx*ny)
    
    # L2 error
    errors = np.abs(decoded_data - reference_data)
    l2_rel_err = np.linalg.norm(errors, axis=0)/np.linalg.norm(reference_data, axis=0)
    l2 = np.mean(l2_rel_err)
    print('Autoencoder relative L2 error: {}\n'.format(round(l2,4)))

    # Mean squared error (MSE)
    mse = mean_squared_error(reference_data, decoded_data)
    print('Mean squared error (MSE): {}\n'.format(round(mse,4)))
    errordir = 'results/d_' + str(args.latent_dim) + '/' + str(args.method) + '/errors_AE/'
    np.savetxt(errordir + 'error_' + str(args.method) + '_seed_' + str(seed_value) + '.txt', 
                                   np.expand_dims(np.array(mse), axis=0), fmt='%e') 

    
    #### Save all data
    # Pass ALL data through the encoder and then split
    x_y_red = autoencoder.encoder(x_y_data_norm.reshape(n*nt+n, nx, ny, 1)).numpy()  # encoder

    if args.ood == 1:
        # Pass all OOD data through the encoder
        x_y_ood_red = autoencoder.encoder(x_y_ood_data_norm.reshape(num_ood*nt+num_ood, nx, ny, 1)).numpy()  # encoder   
    
    if args.noise == 1:
        # Pass noisy input and output data separately through the encoder
        x_noisy_red = autoencoder.encoder(x_noisy_data_norm.reshape(num_noisy, nx, ny, 1)).numpy()
        y_noisy_red = autoencoder.encoder(y_noisy_data_norm.reshape(num_noisy*nt, nx, ny, 1)).numpy()

    # Train-test
    # Split x and y data
    x_red, y_red = x_y_red[:n], x_y_red[n:].reshape(n,nt*args.latent_dim)

    # Split to train/test
    x_train_red, x_test_red = x_red[:num_train], x_red[num_train:]
    y_train_red, y_test_red = y_red[:num_train].reshape(num_train, nt, args.latent_dim), y_red[num_train:].reshape(num_test, nt, args.latent_dim)
    x_train, x_test = inputs[:num_train], inputs[num_train:]
    y_train, y_test = outputs[:num_train], outputs[num_train:] 

    if args.ood == 1:
        # OOD
        x_ood_red, y_ood_red = x_y_ood_red[:num_ood], x_y_ood_red[num_ood:].reshape(num_ood,nt*args.latent_dim)
    
    # Save reduced data (for DeepONet)
    data_dir = 'results/d_' + str(args.latent_dim) + '/' + str(args.method) + '/data/'

    if args.ood == 1 and args.noise == 1:
        np.savez(data_dir + 'data_d_{}.npz'.format(args.latent_dim), latent_dim=args.latent_dim, x_train=x_train, x_test=x_test, 
                                                     y_train=y_train, y_test=y_test, 
                                                     x_train_red=x_train_red, x_test_red=x_test_red, 
                                                     y_train_red=y_train_red, y_test_red=y_test_red,
                                                     x_ood_red=x_ood_red, y_ood_red=y_ood_red, y_ood=outputs_ood,
                                                     x_noisy_red=x_noisy_red, y_noisy_red=y_noisy_red, y_noisy=outputs_n)  

    else:
        np.savez(data_dir + 'data_d_{}.npz'.format(args.latent_dim), latent_dim=args.latent_dim, x_train=x_train, x_test=x_test, 
                                                     y_train=y_train, y_test=y_test, 
                                                     x_train_red=x_train_red, x_test_red=x_test_red, 
                                                     y_train_red=y_train_red, y_test_red=y_test_red)  
        
    # Save autoencoder class (to use decoder later)
    class_AE_dir = 'results/d_' + str(args.latent_dim) + '/' + str(args.method) + '/class_AE/'
    autoencoder.save(class_AE_dir + 'class_AE_{}'.format(args.latent_dim))
    
    
elif args.method == 'WAE':
    
    print('Autoencoder training is completed.')

    plt.figure(figsize=(4,3))
    plt.plot(history_loss_train, linewidth=2)
    plt.plot(history_loss_test, linewidth=2)
    plt.title('Autoencoder loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.yscale('log')
    plt.tight_layout()
    plotdir = 'results/d_' + str(args.latent_dim) + '/' + str(args.method) + '/plots/'
    plt.savefig(plotdir + 'History_both_' + str(args.method) + '_seed_' + str(seed_value) + '.png', dpi=300)
 
    plt.figure(figsize=(4,3))
    plt.plot(history_loss_test, linewidth=2)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['validation'], loc='upper right')
    plt.yscale('log')
    plt.tight_layout()
    plotdir = 'results/d_' + str(args.latent_dim) + '/' + str(args.method) + '/plots/'
    plt.savefig(plotdir + 'History_validation_' + str(args.method) + '_seed_' + str(seed_value) + '.png', dpi=300)
    
    # Compute L2 error on test data
    encoded_data = encoder.forward(tensor_test)
    decoded_data = decoder.forward(encoded_data).detach().numpy().reshape(n_test, nx*ny)
    
    # Compute L2 error on test data
    reference_data = x_y_test.reshape(n_test, nx*ny)

    errors = np.abs(decoded_data - reference_data)
    l2_rel_err = np.linalg.norm(errors, axis=0)/np.linalg.norm(reference_data, axis=0)
    l2 = np.mean(l2_rel_err)
    print('Autoencoder relative L2 error: {}\n'.format(round(l2,4)))
    errordir = 'results/d_' + str(args.latent_dim) + '/' + str(args.method) + '/errors_AE/'
    np.savetxt(errordir + 'error_' + str(args.method) + '_seed_' + str(seed_value) + '.txt', 
                                   np.expand_dims(np.array(l2), axis=0), fmt='%e') 

    
    #### Save all data
    # Pass ALL data through the encoder and then split
    x_y_red = encoder.forward(tensor_all).detach().numpy().reshape(n*nt+n, args.latent_dim)
    
    if args.ood == 1:
        # Pass all OOD data through the encoder
        x_y_ood_red = encoder.forward(tensor_ood).detach().numpy().reshape(num_ood*nt+num_ood, args.latent_dim)
    
    if args.noise == 1:
        # Pass noisy input and output data separately through the encoder
        x_noisy_red = encoder.forward(tensor_x_noisy).detach().numpy().reshape(num_noisy, args.latent_dim)
        y_noisy_red = encoder.forward(tensor_y_noisy).detach().numpy().reshape(num_noisy*nt, args.latent_dim)

    # Train-test
    # Split x and y data
    x_red, y_red = x_y_red[:n], x_y_red[n:].reshape(n,nt*args.latent_dim)

    # Split to train/test
    x_train_red, x_test_red = x_red[:num_train], x_red[num_train:]
    y_train_red, y_test_red = y_red[:num_train].reshape(num_train, nt, args.latent_dim), y_red[num_train:].reshape(num_test, nt, args.latent_dim)
    x_train, x_test = inputs[:num_train], inputs[num_train:]
    y_train, y_test = outputs[:num_train], outputs[num_train:] 

    if args.ood == 1:
        # OOD
        x_ood_red, y_ood_red = x_y_ood_red[:num_ood], x_y_ood_red[num_ood:].reshape(num_ood,nt*args.latent_dim)
    
    # Save reduced data (for DeepONet)
    data_dir = 'results/d_' + str(args.latent_dim) + '/' + str(args.method) + '/data/'

    if args.ood == 1 and args.noise == 1:
        np.savez(data_dir + 'data_d_{}.npz'.format(args.latent_dim), latent_dim=args.latent_dim, x_train=x_train, x_test=x_test, 
                                                     y_train=y_train, y_test=y_test, 
                                                     x_train_red=x_train_red, x_test_red=x_test_red, 
                                                     y_train_red=y_train_red, y_test_red=y_test_red,
                                                     x_ood_red=x_ood_red, y_ood_red=y_ood_red, y_ood=outputs_ood,
                                                     x_noisy_red=x_noisy_red, y_noisy_red=y_noisy_red, y_noisy=outputs_n)  

    else:
        np.savez(data_dir + 'data_d_{}.npz'.format(args.latent_dim), latent_dim=args.latent_dim, x_train=x_train, x_test=x_test, 
                                                     y_train=y_train, y_test=y_test, 
                                                     x_train_red=x_train_red, x_test_red=x_test_red, 
                                                     y_train_red=y_train_red, y_test_red=y_test_red)  

    # Save autoencoder class (to use decoder later)
    class_AE_dir = 'results/d_' + str(args.latent_dim) + '/' + str(args.method) + '/class_AE/'
    torch.save(wae_decoder, class_AE_dir + 'final_decoder.pth')
    
    
    
    
    
    
    
