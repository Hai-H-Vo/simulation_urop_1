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

# from typing import Union, Callable

# DYNAMICS

# @vmap
# def angle_sum(theta1, theta2):
#     init_theta = theta1 + theta2
#     return np.where(
#         init_theta < 0,
#         init_theta + 2 * onp.pi,
#         np.where(init_theta > 2 * onp.pi, init_theta - 2 * onp.pi, init_theta),
#     )

@jit
def angle_correct(theta):
    return np.where(
        theta < 0,
        theta + 2 * onp.pi,
        np.where(theta > 2 * onp.pi, theta - 2 * onp.pi, theta),
    )

# CHIRAL PARTICLES

@vmap
def normal(v, theta):
  return np.array([v * np.cos(theta), v * np.sin(theta)])

@jit
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

# PEDESTRIANS

def init_velocity(key, speed_mean, N):
    """
    Initializes particles with 2D Maxwell-Boltzmann distrib, with given mean speed
    """
    return np.sqrt(2/ onp.pi) * speed_mean * random.normal(key, shape=(N, 2))

@jit
def ttc_potential_fn(k, t, t_0):
   return (k / np.square(t)) * np.exp(-t/ t_0)

@jit
def time_to_collide(dpos, V_i, V_j, R):
   """
   dpos = pos_i - pos_j, V_i, V_j : ndarrays
   R : float

   The time to collide is the smaller root of the
   following quadratic equation at^2 + bt + c = 0
   """
   # stop grad for non-positional args
   V_i = lax.stop_gradient(V_i)
   V_j = lax.stop_gradient(V_j)
   R = lax.stop_gradient(R)

   t_max = 9
   dv = V_i - V_j

   a = np.dot(dv, dv)
   b = - np.dot(dpos, dv)
   c = np.dot(dpos, dpos) - np.square(2 * R)

   det = np.square(b) - a * c

   t1 = np.where(det > 0, (-b - np.sqrt(det))/(a), t_max)
   t2 = np.where(det > 0, (-b + np.sqrt(det))/(a), t_max)

   t1 = np.where(t1 < 0, np.where(t2 > 0, 0.1, t_max), t1)  # maybe should be 0?

   t1 = np.where(t1 > t_max, t_max, t1)

   return t1

def ttc_tot(pos, V, R, displacement):
   ttc = vmap(vmap(time_to_collide, (0, None, 0, None)), (0, 0, None, None))

   dpos = space.map_product(displacement)(pos, pos)

   return ttc(dpos, V, V, R)

def ttc_potential_tot(pos, V, R, displacement, k=1.5, t_0=3.0):
   """
   The potential energy of pedestrian interaction, according to
   the anticipatory interaction law detailed in [1].

   Inputs:
      pos (ndarray)     : position vector of all particles
      V (ndarray)       : velocity vector "   "      "
      R (float)         : collision radius of a particle
      displacement (fn) : displacement function produced by jax_md.space
      k, t_0 (floats)   : interaction params

   Output:
      ndarray of potential energy of each particle

   [1] I. Karamouzas, B. Skinner, Stephen J. Guy.
   "Universal Power Law Governing Pedestrian Interactions"
   """
   return np.sum(ttc_potential_fn(k, ttc_tot(pos, V, R, displacement), t_0)) / 2
