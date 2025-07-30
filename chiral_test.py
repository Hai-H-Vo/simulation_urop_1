from simulator import dynamics, render
from jax import random
from jax_md import space
from jax import lax
import numpy as onp

N = 1000
dt = 0.0176
delta = 25
box_size = 100

# Create RNG state to draw random numbers (see LINK).
rng = random.PRNGKey(0)

displacement, shift = space.periodic(box_size)

init, step = dynamics.chiral_am(shift, displacement, dt, N, box_size)

state = init(rng)
positions = []
thetas = []

for i in range(100):
  state = lax.fori_loop(0, delta, step, state)
  positions += [state.position]
  thetas += [state.theta]

render.render(box_size, positions, dt, delta, 'chiral_test_2', extra=thetas, limits=(0, 2 * onp.pi))
