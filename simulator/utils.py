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

# SPACES

# def closed_box(box_size):
#    def displacement_fn(R_1, R_2, **unused_kwargs):
#       dR = R_1 - R_2
#       return dR

#    def shift_fn(R, dR, **unused_kwargs):
#       return R + dR

#    return displacement_fn, shift_fn

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

# @jit
# def normalize_cap(v, v_lim=0):
#     v_norm = np.linalg.norm(v, axis=1, keepdims=True)
#     return np.where(v_norm > v_lim, v/v_norm, v)

def _normalize_cap(v, v_lim=0):
    v_norm = np.linalg.norm(v, keepdims=True)
    return np.where(v_norm > v_lim, v/v_norm, v)

@jit
def normalize_cap(v, v_lim=0):
   return vmap(_normalize_cap, (0, None))(v, v_lim)

# @partial(jit, static_argnums=1)
def wall_dist(pos, start, end, displacement):
   """
   Computes displacement from a particle to a wall. A wall is parameterized by
   its two endpoints.

   Inputs:
      pos (Array): array denoting particle position
      start, end (Array): arrays denoting wall positions
      displacement (fn): function to compute distance between 2 points

   Output:
      Array denoting displacement from particle to wall
   """
   wall_len = np.dot(end - start, end - start)
   # t = max(0, min(1, np.dot(pos - start, end - start) / wall_len))
   t = np.max(np.array([0, np.min(np.array([1, np.dot(pos - start, end - start) / wall_len]), axis = 0)]), axis = 0)
   proj = start + t * (end - start)
   return pos - proj

# @partial(jit, static_argnums=1)
def wall_energy(pos, wall, radius, displacement):
   """
   Used to model the repulsion between a particle and a wall.

   Inputs:
      pos (Array): particle position
      wall (Wall obj): object representing a wall, with start and end attrs

   Output:
      Interaction between particle and the wall
   """
   start = lax.stop_gradient(wall.start)
   end = lax.stop_gradient(wall.end)

   dist = wall_dist(pos, start, end, displacement)

   # return 5 * radius / (np.linalg.norm(dist) - radius)
   return 1 / (3 * (np.linalg.norm(dist) / radius) ** 3)

def wall_energy_tot(poss, wall, radius, displacement):
   return np.sum(vmap(wall_energy, (0, None, None, None))(poss, wall, radius, displacement))

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

def init_goal_speed(key, N, speed_mean=1.3, speed_var=0.3):
    """
    Initializes pedestrians with their own preferred walking speeds.
    The preferred speed is normally distributed around speed_mean, with
    s.d speed_var, and truncated to fall into [speed_mean - speed_var, speed_mean + speed_var]
    """
    return speed_mean + speed_var * random.truncated_normal(key,
                                                            lower=-speed_var,
                                                            upper=+speed_var,
                                                            shape=(N,))

@jit
def ttc_potential_fn(k, t, t_0):
   return np.array((k / np.square(t)) * np.exp(-t/ t_0))


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

   t_max = 99
   dv = V_i - V_j

   a = np.dot(dv, dv)
   b = - np.dot(dpos, dv)
   c = np.dot(dpos, dpos) - np.square(2 * R)

   det = np.square(b) - a * c

   t1 = np.where(det > 0, np.nan_to_num((-b - np.sqrt(det))/(a)), t_max)
   t2 = np.where(det > 0, np.nan_to_num((-b + np.sqrt(det))/(a)) , t_max)

   t1 = np.where(t1 < 0, np.where(t2 > 0, 0.1, t_max), t1)  # maybe should be 0?

   t1 = np.where(t1 > t_max, t_max, t1)

   return t1

@jit
def ttc_force(dpos, V_i, V_j, R, k, t_0):
   t_max = 99
   dv = V_i - V_j

   sq_dist = np.dot(dpos, dpos)
   sq_rad = np.square(2 * R)
   adjusted_sq_rad = np.where(sq_rad > sq_dist, .99 * sq_dist, sq_rad)

   a = np.dot(dv, dv)
   b = - np.dot(dpos, dv)
   c = sq_dist - adjusted_sq_rad
   # c = np.dot(dpos, dpos) - np.square(2 * R)

   det = np.square(b) - a * c

   t = (b - np.sqrt(det)) / a

   return np.where(np.logical_or(det < 0, np.logical_and(a < 0.001, a > -0.001)),
                   np.array([0, 0]),
                   np.where(np.logical_or(t < 0, t > t_max),
                            np.array([0, 0]),
                            - k*np.exp(-t/t_0)*(dv - (dv * b - dpos * a)/(np.sqrt(det)))/(a*np.square(t))*(2/t+ 1/t_0)))

@jit
def safe_time_to_collide(dpos, V_i, V_j, R, i, j):
   return np.where(i != j, time_to_collide(dpos, V_i, V_j, R), 99)

def ttc_tot(pos, V, R, displacement):
   ttc = vmap(vmap(
      safe_time_to_collide, (0, None, 0, None, None, 0)
      ), (0, 0, None, None, 0, None)
      )
   dpos = space.map_product(displacement)(pos, pos)

   dim = V.shape[0]

   I = np.arange(0, dim)
   J = np.arange(0, dim)

   return ttc(dpos, V, V, R, I, J)

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
   return np.array(np.sum(smap._diagonal_mask(ttc_potential_fn(k, ttc_tot(pos, V, R, displacement), t_0))) / 2)

def ttc_force_tot(pos, V, R, displacement, k=1.5, t_0=3.0):
   force_fn = vmap(vmap(ttc_force, (0, 0, None, None, None, None)), (0, None, 0, None, None, None))

   dpos = space.map_product(displacement)(pos, pos)

   return np.sum(normalize_cap(force_fn(dpos, V, V, R, k, t_0), 5), axis=1)
   # return force_fn(dpos, V, V, R, k, t_0)
   # return np.sum(force_fn(dpos, V, V, R, k, t_0), axis = 1)

def goal_velocity_force(state):
   if state.goal_orientation is None:
      return (normal(state.goal_speed, state.orientation()) - state.velocity) / .5
   return (normal(state.goal_speed, state.goal_orientation) - state.velocity) / .5
