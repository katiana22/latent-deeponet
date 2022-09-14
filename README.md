## Table of contents
* [General info](#general-info)
* [Methods-pipeline](#methods-pipeline)
* [Examples](#examples)
* [Contents](#contents)
* [Getting started](#getting-started)

## General info

This Git repository contains python codes for implementing the Latent DeepONet model proposed here: 
Latent DeepONet leverages auroencoder models with the operator regressor DeepONet, to learn operator on latent spaces. The code can be used to perform regression on very high-dimensional PDE problems and generate comparative results for multiple autoencoder architectures and latent dimensions. As demonstrated in the associated paper, latent DeepONet results in higher predictive accuracy that standard DeepONet which is trained on the full dimensional data.

## Methods-pipeline

Details of the methdology can be found in the published paper here.

Below, a **graphical summary** of the method is provided:

<!---
<img src="pipeline.png" width="700">

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
conda create -n ld_env python==3.8  
conda activate ld_env
```

**2.** Clone our repo:

```
git clone https://github.com/katiana22/latent-deeponet.git
```

**3.** Install dependencies via ```pip''' with the following commands: 

```
cd latent-deeponet 
pip install -r requirements.txt
``` 

or  

Install the entire conda environment:

```
conda env create -f environment.yml
```

## Citation

### Mainteners
[Katiana Kontolati](https://katiana22.github.io/)

:email: : kontolati@jhu.edu



