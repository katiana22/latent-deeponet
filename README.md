## Table of contents
* [General info](#general-info)
* [Methods-pipeline](#methods-pipeline)
* [Examples](#examples)
* [Contents](#contents)
* [Getting started](#getting-started)
* [Demonstration](#demonstration)

## General info

This Git repository contains python codes for implementing the Latent DeepONet model proposed here:   
Latent DeepONet leverages auroencoder models with the operator regressor DeepONet, to learn operators on latent spaces by leveraging the intrinsic dimensionality of physics-based data. 

The code has the following capabilities:  
* Perform operator regression for PDE problems of very high dimensionality (e.g., high-dimensional time-dependent PDEs)
* Compare the performance of latent DeepONet for different autoencoder models. The current version includes the following models:   
    1. **Autoencoder** (vanilla-AE)  
    2. **Multi-Layer Autoencoder** (MLAE)
    3. **Convolutional Autoencoder** (CAE)  
    4. **Variational Autoencoder** (VAE) and
    5. **Wasserstein Autoencoder** (WAE)
* Train latent DeepONet for different values of the latent dimension (*d*) and compare results with the standard DeepONet trained on the full-dimensional data.

## Methods-pipeline

Details of the methdology can be found in the published paper here.

Below, a **graphical summary** of the method is provided:

<p align="center">
    <img width="800" src="schematic.png" alt="Latent DeepONet method">
</p>

<!---
## Application

Three illustrative examples are provided. The first considers a dielectric cylinder suspended in a homogeneous electric field. The second is the classic Lotka-Volterra dynamical system modeling the evolution of two species interacting with each other, one a predator and one a prey. Finally, the third example considers a system of advection-diffusion-reaction equations which models a first-order chemical reaction between two species. 
 
<img src="applications.png" width="900">
--->

## Contents

* ```AE.py``` - Contains all AE classes and code to train Autoencoders

* ```DON.py``` - Containts all DON classes and code to train latent DeepONet

* ```plot.py``` - Generates plots with comparative results for all methods and latent dimensions

* ```main.ipynb``` - Demonstrates how to generate results for multiple methods and latent dimensions

* ```driver.sh``` - Demonstrates how to run the scripts on the terminal


## Getting started

**1.** Create an Anaconda Python 3.8 virtual environment:
```
conda create -n ld_env python==3.8.13  
conda activate ld_env
```

**2.** Clone our repo:

```
git clone https://github.com/katiana22/latent-deeponet.git
```

**3.** Install dependencies via ```pip``` with the following commands: 

```
cd latent-deeponet 
pip install -r requirements.txt
``` 

or  

Install the entire conda environment:

```
conda env create -f environment.yml
```

## Demonstration  

## Citation  

### Mainteners
[Katiana Kontolati](https://katiana22.github.io/)

:email: : kontolati@jhu.edu



