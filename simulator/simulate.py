# SIMULATION
from .dynamics import dynamics
from .render import render
from jax import lax

def simulate(SIMUL_STEPS, DELTA, update, state):
    particle_buffer = [state['particle']]

    for _ in range(SIMUL_STEPS):
      state = lax.fori_loop(0, DELTA, update, state)
      particle_buffer += [state['particle']]

    return particle_buffer
