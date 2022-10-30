# make sure to $ conda activate dedalus

import numpy as np
import dedalus.public as d3
import logging
import h5py, os
logger = logging.getLogger(__name__)
import matplotlib
import subprocess, glob

# Initialize lists
init_perturbs, outputs = [], []
n_samples = 150

# Simulation units
meter = 1 / 6.37122e6
hour = 1
second = hour / 3600

# Parameters
Nphi = 256
Ntheta = 256
dealias = 3/2
R = 6.37122e6 * meter
Omega = 7.292e-5 / second
nu = 1e5 * meter**2 / second / 32**2 # Hyperdiffusion matched at ell=32
g = 9.80616 * meter / second**2
H = 1e4 * meter
timestep = 600 * second
stop_sim_time = 360 * hour # control time of simulation
dtype = np.float64

# draw random samples for alpha/beta
alphas = np.random.uniform(1/9, 1/2, n_samples)
betas = np.random.uniform(1/5, 1/30, n_samples)
params = (alphas, betas)

for i in range(n_samples):
    
    print('Iteration:', i+1)
    print('alpha: {}, beta: {}'.format(alphas[i], betas[i]))
    print('')

    # Bases
    coords = d3.S2Coordinates('phi', 'theta')
    dist = d3.Distributor(coords, dtype=dtype)
    basis = d3.SphereBasis(coords, (Nphi, Ntheta), radius=R, dealias=dealias, dtype=dtype)

    # Fields
    u = dist.VectorField(coords, name='u', bases=basis)
    h = dist.Field(name='h', bases=basis)

    # Substitutions
    zcross = lambda A: d3.MulCosine(d3.skew(A))

    # Initial conditions: zonal jet
    phi, theta = dist.local_grids(basis)
    lat = np.pi / 2 - theta + 0*phi
    umax = 80 * meter / second
    lat0 = np.pi / 7
    lat1 = np.pi / 2 - lat0
    en = np.exp(-4 / (lat1 - lat0)**2)
    jet = (lat0 <= lat) * (lat <= lat1) # True/False with the zone of the flow
    u_jet = umax / en * np.exp(1 / (lat[jet] - lat0) / (lat[jet] - lat1))
    u['g'][0][jet]  = u_jet

    # Initial conditions: balanced height
    c = dist.Field(name='c')
    problem = d3.LBVP([h, c], namespace=locals())
    problem.add_equation("g*lap(h) + c = - div(u@grad(u) + 2*Omega*zcross(u))")
    problem.add_equation("ave(h) = 0")
    solver = problem.build_solver()
    solver.solve()

    # Initial conditions: perturbation
    lat2 = np.pi / 4
    hpert = 120 * meter
    alpha = alphas[i]
    beta = betas[i]

    print(alpha, beta)
    h['g'] += hpert * np.cos(lat) * np.exp(-(phi/alpha)**2) * np.exp(-((lat2-lat)/beta)**2)
    init_perturbs.append(h['g'])

    # Problem
    problem = d3.IVP([u, h], namespace=locals())
    problem.add_equation("dt(u) + nu*lap(lap(u)) + g*grad(h) + 2*Omega*zcross(u) = - u@grad(u)")
    problem.add_equation("dt(h) + nu*lap(lap(h)) + H*div(u) = - div(h*u)")

    # Solver
    solver = problem.build_solver(d3.RK222)
    solver.stop_sim_time = stop_sim_time

    # Analysis
    snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=1*hour, max_writes=500)
    snapshots.add_task(h, name='height')
    snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')

    # Main loop
    try:
        logger.info('Starting main loop')
        while solver.proceed:
            solver.step(timestep)
            #if (solver.iteration-1) % 10 == 0:
            #    logger.info('Iteration=%i, Time=%e, dt=%e' %(solver.iteration, solver.sim_time, timestep))
    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise
    finally:
        solver.log_stats()

    # extract output from h5 file
    filename = './snapshots/snapshots_s1.h5'
    hf = h5py.File(filename, 'r')
    output = np.array(hf['tasks']['vorticity'])[::5]
    outputs.append(output)

    init_perturbs_save = np.array(init_perturbs)
    outputs_save = np.array(outputs)

    # save updated results
    np.savez('../../scr16_mshiel10/kontolati/shallow/data/shallow-water-1.npz', params=params, inputs=init_perturbs_save, outputs=outputs_save)


