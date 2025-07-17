# IMPORTS

import numpy as onp
import jax.numpy as np
from jax import random
from jax import jit
from jax import vmap
from jax import lax
from jax import config
config.update("jax_enable_x64", True)
config.update('jax_debug_nans', True)

from jax_md import space, smap, energy, minimize, quantity, simulate, partition, util
from jax_md.util import f32

from collections import namedtuple

vectorize = np.vectorize

from functools import partial
from simulator.utils import angle_correct, normal, align_tot, ttc_potential_tot
from simulator.render import render
from simulator.dynamics import pedestrian


N = 2
dt = 0.01
delta = 25
box_size = 10
key = random.PRNGKey(0)
displacement, shift = space.periodic(box_size)

V_key, pos_key = random.split(key)

def energy_fn(state):
    return ttc_potential_tot(state.position, state.velocity, state.radius, displacement, 1.5, 3)

init, step = pedestrian(shift, energy_fn, dt, N)

# pos = box_size * random.uniform(pos_key, (N, 2))

pos = np.array([[0.0, 0.0],
                [3.0, 0.0]])

velocity = np.array([[-1.0, 0.0],
                     [0.0, 0.0]])

print(pos)

# state = init(pos, 1.0, key=V_key, speed_mean=1.3)
state = init(pos, 1.0, velocity=velocity)

print(state.velocity)

positions = []
thetas = []

state = step(0, state)

# for i in range(100):
#   state = lax.fori_loop(0, delta, step, state)
#   positions += [state.position]
#   thetas += [state.orientation()]

# render(box_size, positions, dt, delta, 'pedestrian_test', extra=thetas, limits=(0, 2 * onp.pi))
