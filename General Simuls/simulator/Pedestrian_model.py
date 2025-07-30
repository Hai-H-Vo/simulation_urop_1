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
config.update('jax_traceback_filtering', 'off')

from jax_md import space, smap, energy, minimize, quantity, simulate, partition, util
from jax_md.util import f32

from collections import namedtuple

vectorize = np.vectorize

from functools import partial
from utils import angle_correct, normal, align_tot, time_to_collide, ttc_tot, ttc_potential_fn, ttc_potential_tot, ttc_force_tot, wall_energy, wall_energy_tot
from render import render
from dynamics import pedestrian, PedestrianState, StraightWall


N = 1000
dt = 0.01
delta = 25
box_size = 10
key = random.PRNGKey(0)
displacement, shift = space.free()

V_key, pos_key = random.split(key)

ll = np.array([0., 0.])
ul = np.array([0., box_size])
lr = np.array([box_size, 0.])
ur = np.array([box_size, box_size])
wall1 = StraightWall(ll, ul)
wall2 = StraightWall(ll, lr)
wall3 = StraightWall(ul, ur)
wall4 = StraightWall(lr, ur)

# def energy_fn(state):
#     return ttc_potential_tot(state.position, state.velocity, state.radius, displacement, 1.5, 3)

# def energy_fn(state):
#     dR = space.map_product(displacement)(state.position, state.position)

#     spring_tot = vmap(partial(energy.simple_spring, length=0.5), (1))

#     return np.sum(spring_tot(dR))/2

# def energy_fn(state):
#     dR = space.map_product(displacement)(state.position, state.position)

#     lennard_tot = vmap(energy.lennard_jones, (1))

#     return np.sum(lennard_tot(dR)) / 2

# def energy_fn(state):
#     return energy.lennard_jones_pair(displacement)(state.position)

# def force_fn(state):
#     return quantity.canonicalize_force(energy_fn)(state)

# def energy_fn(pos):
#     ll = np.array([0., 0.])
#     ul = np.array([0., box_size])
#     lr = np.array([box_size, 0.])
#     ur = np.array([box_size, box_size])
#     wall1 = StraightWall(ll, ul)
#     wall2 = StraightWall(ll, lr)
#     wall3 = StraightWall(ul, ur)
#     wall4 = StraightWall(lr, ur)
#     return wall_energy_tot(pos, wall1, displacement) + wall_energy_tot(pos, wall2, displacement) + wall_energy_tot(pos, wall3, displacement) + wall_energy_tot(pos, wall4, displacement)

def force_fn(state):
#     wall_force = quantity.force(energy_fn)
#     return PedestrianState(ttc_force_tot(state.position, state.velocity, state.radius, displacement, 1.5, 3) + wall_force(state.position), None, None, None)
    return ttc_force_tot(state.position, state.velocity, state.radius, displacement, 1.5, 3)
# init, step = pedestrian(shift, energy_fn, dt, N)

init, step = pedestrian(shift, force_fn, dt, N)

pos = box_size * random.uniform(pos_key, (N, 2))

pos = np.array([[0.0, 0.0],
                [3.0, 0.0],
                ])

# pos = np.array([0., 0.])

# print(wall_energy_tot(pos, wall4, displacement))

velocity = np.array([[1.0, 0.0],
                     [0.0, 0.0]])

state = init(pos, 1.0, velocity=velocity)

# print(ttc_tot(pos, velocity, 1.0, displacement))

# print(ttc_potential_fn(1.5, ttc_tot(pos, velocity, 1.0, displacement), 3.0))

# print(smap._diagonal_mask(ttc_potential_fn(1.5, ttc_tot(pos, velocity, 1.0, displacement), 3.0)))

# print(energy_fn(state))

# print(quantity.force(energy_fn)(state))


print(normal(state.speed(), state.orientation()))


# print(pos)

# print(space.map_product(displacement)(pos, pos))

# state = init(pos, 1.0, key=V_key, speed_mean=1.3)


# print(state.velocity)

positions = []
thetas = []

# state = step(0, state)

# print(state)

# for i in range(100):
#   state = lax.fori_loop(0, delta, step, state)
#   positions += [state.position]
#   thetas += [state.orientation()]

# print(state)

# render(box_size, positions, dt, delta, 'pedestrian_test', extra=thetas, limits=(0, 2 * onp.pi))
