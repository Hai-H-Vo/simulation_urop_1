# IMPORTS

import numpy as onp
import jax.numpy as np
from jax import random
from jax import jit
from jax import vmap
from jax import lax
from jax import config
config.update("jax_enable_x64", True)

from jax_md import space, smap, energy, minimize, quantity, simulate, partition, util
from jax_md.util import f32

from collections import namedtuple

vectorize = np.vectorize

from functools import partial

# DYNAMICS

@vmap
def angle_sum(theta1, theta2):
    init_theta = theta1 + theta2
    return np.where(
        init_theta < 0,
        init_theta + 2 * onp.pi,
        np.where(init_theta > 2 * onp.pi, init_theta - 2 * onp.pi, init_theta),
    )

# CHIRAL PARTICLES

@vmap
def normal(v, theta):
  return np.array([v * np.cos(theta), v * np.sin(theta)])

def align_fn(dr, theta_i, theta_j):
   align_spd = np.sin(theta_j - theta_i)
   dR = space.distance(dr)
   return np.where(dR < 1., align_spd, 0.)

def align_tot(R, theta, displacement):
   # Alignment factor
   align = vmap(vmap(align_fn, (0, None, 0)), (0, 0, None))

   # Displacement between all points
   dR = space.map_product(displacement)(R, R)

   return np.sum(align(dR, theta, theta), axis=1)

class ChiralAMState(namedtuple('ChiralState', ['position', 'theta', 'speed', 'omega', 'key'])):
    pass

def chiral_am(shift_fn, displacement_fn, dt, N, box_size, g = 0.018, D_r = 0.009):
    """
    Simulation of chiral active particle dynamics.

    The default values of the simulation function are provided in [1].

    Inputs:
        shift_fn (func): A function that displaces positions, `R`, by an amount `dR`.
            Both `R` and `dR` should be ndarrays of shape `[n, spatial_dimension]`.
        dt (float): Floating point number specifying the timescale (step size) of the
            simulation.
        N (int): Integer number specifying the number of particles in the simulation
        box_size (float): Floating point number specifying the length of the
            square box simulation.
        g (float): Floating point number specifying the strength of the polar alignment
            interactions
        D_r (float): Floating point number specifying the rotational diffusion constant

    Outputs:
        init_fn, step_fn (funcs): functions to initialize the simulation and to timestep
            the simulation

    [1] R. Supekar, B. Song, A. Hastewell. "Learning hydrodynamic equations for active matter from particle
        simulations and experiments".
    """
    def init_fn(key, mu_v = 1.0, sigma_v = 0.4, mu_r = 2.2, sigma_r = 1.7, cutoff_omega = 1.4):
        """
        Initializes speed v and rotational radius r in a Gaussian distribution.
        Angular velocity omega is determined by the relation v = omega * r
        """
        key, v_key, omega_key, R_key, theta_key = random.split(key, 5)

        # SPEED AND ANGULAR VELOCITY INIT
        # sample from gaussian distrib + filter for positive vals
        spd = mu_v + sigma_v * random.normal(v_key, (10 * N, ))
        r = mu_r + sigma_r * random.normal(omega_key, (10 * N, ))

        condition1 = np.where(spd > 0)[0]
        spd = spd[condition1]
        r = r[condition1]

        condition2 = np.where(r > 0)[0]
        spd = spd[condition2]
        r = r[condition2]
        omega = np.divide(spd, r)

        condition3 = np.where(omega <= cutoff_omega)[0]
        spd = spd[condition3]
        omega = omega[condition3]

        spd = spd[:N]
        omega = omega[:N]

        # COORDS INIT
        R = box_size * random.uniform(R_key, (N, 2))
        theta = random.uniform(theta_key, (N,), maxval=2*onp.pi)

        state = ChiralAMState(R, theta, spd, omega, key)

        return state

    @jit
    def step_fn(_, state):
        # Forward-Euler scheme
        n = normal(state.speed, state.theta)
        dR = n * dt
        dtheta = (state.omega + g * align_tot(state.position, state.theta, displacement_fn)) * dt

        # Stochastic step:
        key, split = random.split(state.key)
        stheta = np.sqrt(2 * D_r * dt) * random.normal(split, state.theta.shape)

        return ChiralAMState(shift_fn(state.position, dR),
                             angle_sum(state.theta, dtheta + stheta),
                             state.speed,
                             state.omega,
                             key)

    return init_fn, step_fn
