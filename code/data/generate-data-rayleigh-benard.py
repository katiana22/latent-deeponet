import numpy as np
import dedalus.public as d3
import logging
import h5py, os
logger = logging.getLogger(__name__)
from matplotlib import pylab as plt

# Load input data
file = np.load('./inputs_KLE_lx_5_ly_8_v_4.npz') # original
#file2 = np.load('./inputs_KLE_lx_0.25_ly_0.3_v_0.15.npz') # smaller length scale (for Nature Comms reviews)
file2 = np.load('./inputs_KLE_lx_0.6_ly_0.7_v_0.15.npz') # higher length scale (for Nature Comms reviews)

samples = file2['inputs'] # choose file two for the RBF that is added on top of the linear profile
n_samples = samples.shape[0] # Num of samples
print('Number of total samples:', n_samples)

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

samples_norm = []
for i in range(n_samples):
    samples_norm.append(NormalizeData(samples[i,:,:]))
samples_norm = np.array(samples_norm)

#mean_sample = samples_norm[0,:,:]
mean_sample = NormalizeData(file['inputs'][0,:,:])

final_samples = [mean_sample]
for i in range(n_samples-1):
    #sign = 1 if random.random() < 0.5 else -1
    sign = 1
    final_samples.append(mean_sample + sign*0.2*samples_norm[i+1])
final_samples = np.array(final_samples)

final_outputs = []
for k in range(n_samples):

    print('Iteration:', k+1)
    print('')
    
    # Parameters
    Lx, Lz = 4, 1
    Nx, Nz = 128, 128
    idx = 18 # choose which random sample to run

    x_ = np.linspace(0, Lx, Nx).T
    z_ = np.linspace(0, Lz, Nz).T
    xx, zz = np.meshgrid(x_, z_)

    Rayleigh = 2e6
    Prandtl = 1
    dealias = 3/2
    stop_sim_time = 50
    timestepper = d3.RK222
    max_timestep = 0.125
    dtype = np.float64

    # Bases
    coords = d3.CartesianCoordinates('x', 'z')
    dist = d3.Distributor(coords, dtype=dtype)
    xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
    zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)

    # Fields
    p = dist.Field(name='p', bases=(xbasis,zbasis))
    b = dist.Field(name='b', bases=(xbasis,zbasis))
    u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis))
    tau_p = dist.Field(name='tau_p')
    tau_b1 = dist.Field(name='tau_b1', bases=xbasis)
    tau_b2 = dist.Field(name='tau_b2', bases=xbasis)
    tau_u1 = dist.VectorField(coords, name='tau_u1', bases=xbasis)
    tau_u2 = dist.VectorField(coords, name='tau_u2', bases=xbasis)

    # Substitutions
    kappa = (Rayleigh * Prandtl)**(-1/2)
    nu = (Rayleigh / Prandtl)**(-1/2)
    x, z = dist.local_grids(xbasis, zbasis)
    ex, ez = coords.unit_vector_fields(dist)
    lift_basis = zbasis.derivative_basis(1)
    lift = lambda A: d3.Lift(A, lift_basis, -1)
    grad_u = d3.grad(u) + ez*lift(tau_u1) # First-order reduction
    grad_b = d3.grad(b) + ez*lift(tau_b1) # First-order reduction

    # Problem
    # First-order form: "div(f)" becomes "trace(grad_f)"
    # First-order form: "lap(f)" becomes "div(grad_f)"
    problem = d3.IVP([p, b, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2], namespace=locals())
    problem.add_equation("trace(grad_u) + tau_p = 0")
    problem.add_equation("dt(b) - kappa*div(grad_b) + lift(tau_b2) = - u@grad(b)")
    problem.add_equation("dt(u) - nu*div(grad_u) + grad(p) - b*ez + lift(tau_u2) = - u@grad(u)")
    problem.add_equation("b(z=0) = Lz")
    problem.add_equation("u(z=0) = 0")
    problem.add_equation("b(z=Lz) = 0")
    problem.add_equation("u(z=Lz) = 0")
    problem.add_equation("integ(p) = 0") # Pressure gauge

    # Solver
    solver = problem.build_solver(timestepper)
    solver.stop_sim_time = stop_sim_time

    # Initial conditions
    b.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
    b['g'] *= z * (Lz - z) # Damp noise at walls
    b['g'] += Lz - z # Add linear background
    b['g'] = final_samples[k,:,:] # random input

    # Analysis
    snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.25, max_writes=200)
    snapshots.add_task(b, name='buoyancy')
    snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')

    # CFL
    CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.5, threshold=0.05,
                 max_change=1.5, min_change=0.5, max_dt=max_timestep)
    CFL.add_velocity(u)

    # Flow properties
    flow = d3.GlobalFlowProperty(solver, cadence=10)
    flow.add_property(np.sqrt(u@u)/nu, name='Re')

    # Main loop
    startup_iter = 10
    try:
        logger.info('Starting main loop')
        while solver.proceed:
            timestep = CFL.compute_timestep()
            solver.step(timestep)
            if (solver.iteration-1) % 10 == 0:
                max_Re = flow.max('Re')
                logger.info('Iteration=%i, Time=%e, dt=%e, max(Re)=%f' %(solver.iteration, solver.sim_time, timestep, max_Re))
    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise
    finally:
        solver.log_stats()

    # Load and save all snapshots
    hf = h5py.File('./snapshots/snapshots_s1.h5', 'r')
    output = np.array(hf['tasks']['buoyancy'])[::2,:,:]
    final_outputs.append(output)
    
if not os.path.exists('data'):
    os.makedirs('data')    
       
final_outputs = np.array(final_outputs)
datafile = 'Benard_data_nx_128_nz_128_nt_100.npz' 
datadir = './data/'
np.savez(os.path.join(datadir, datafile), n_samples=n_samples, inputs=final_samples, outputs=final_outputs)






