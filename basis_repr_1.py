# IMPORTS

import numpy as onp
import jax.numpy as np
import optax
from jax import random
from jax import jit
from jax import vmap
from jax import grad
from jax import lax
from jax import config
config.update("jax_enable_x64", True)
config.update('jax_debug_nans', True)
config.update('jax_traceback_filtering', 'off')

from jax_md import space, smap, energy, minimize, quantity, simulate, partition, util
from jax_md.util import f32

from collections import namedtuple

vectorize = np.vectorize

from functools import partial
from simulator.utils import ttc_force, _normalize
from simulator.force import general_force_generator
from simulator.render import render
from simulator.dynamics import pedestrian, PedestrianState, StraightWall

# POSITIONS AND VELOCITIES
POS_NUMS = 2
ANGLE_NUMS = 2
R = 0.1
K = 1.5
T_0 = 3.0
V_2_MAG = 1.
CYCLE_NUMS = 1000

positions = np.stack([np.linspace(0.3, 2., POS_NUMS), np.zeros([POS_NUMS,])], axis=1)
angles = np.linspace(0, 1/2, ANGLE_NUMS)
v_2 = V_2_MAG * np.stack([np.cos(onp.pi * angles), np.sin(onp.pi * angles)], axis=1)
v_1 = np.zeros([ANGLE_NUMS, 2])

# PARAMS INIT
paral_weights = np.zeros([10, 10, 10])
perpen_weights = np.zeros([10, 10, 10])
d_0 = 10.
v_0 = 10.

def _loss_fn(params, pos, v1, v2):
    paral_weights = params['paral']
    perpen_weights = params['perpen']
    v_0 = params['v0']
    d_0 = params['d0']
    return np.linalg.norm(general_force_generator(paral_weights, perpen_weights, v_0, d_0)(pos, v1, v2) - ttc_force(pos, v1, v2, R, K, T_0)) ** 2

def loss_fn(params, pos, v1, v2):
    # loss_fn = sum over sets of (pos, v) ||F_pred - F||^2
    full_loss_fn = vmap(vmap(_loss_fn, (None, None, 0, 0)), (None, 0, None, None))
    return np.sum(full_loss_fn(params, pos, v1, v2))

# OPTIMIZATION
start_learning_rate = 0.1
optimizer = optax.adam(start_learning_rate)

# PARAMS
params = {'paral' : paral_weights,
          'perpen' : perpen_weights,
          'd0' : d_0,
          'v0' : v_0}
opt_state = optimizer.init(params)

# UPDATE LOOP
for i in range(CYCLE_NUMS):
  print(f"Current update loop: {i}/{CYCLE_NUMS}")
  grads = grad(loss_fn)(params, positions, v_1, v_2)
  # fails before this point
  updates, opt_state = optimizer.update(grads, opt_state)
  params = optax.apply_updates(params, updates)

# allow_int = False:
# TypeError: grad requires real- or complex-valued inputs (input dtype that is a sub-dtype of np.inexact), but got int64.
# If you want to use Boolean- or integer-valued inputs, use vjp or set allow_int to True.

# allow_int = True:
# FloatingPointError: invalid value (nan) encountered in mul
# When differentiating the code at the top of the callstack:
# /Users/vohoanghai/Documents/MIT/UROP_1/Work/General Simuls/simulator/force.py:105:15 (general_force_generator.<locals>.general_force)
# /Users/vohoanghai/Documents/MIT/UROP_1/Work/General Simuls/basis_repr_2.py:55:11 (_loss_fn)
