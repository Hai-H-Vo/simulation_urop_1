# SIMULATION PARAMS
box_size = 100      # float specifying side length of box
N = 12000           # number of particles in our system
SIMUL_STEPS = 2275  # number of simulation saves performed
DT = 0.0176         # simulation time step
DELTA = 25          # number of time steps between simulation save

# IMPORTS
import numpy as onp

from jax import config ; config.update('jax_enable_x64', True)
import jax.numpy as np
from jax import random
from jax import jit
from jax import vmap
from jax import lax

vectorize = np.vectorize

from functools import partial

from collections import namedtuple
import base64

import IPython
from IPython.display import HTML, display
import time

import os

from jax_md import space, smap, energy, minimize, quantity, simulate, partition, util
from jax_md.util import f32

import ffmpeg

# Plotting

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import seaborn as sns

# Tell Matplotlib how to embed animations
plt.rcParams['animation.html'] = 'jshtml'      # or 'html5'

sns.set_style(style='white')

normalize = lambda v: v / np.linalg.norm(v, axis=1, keepdims=True)

# RENDERING

def render(box_size, states, name="supekar"):
  """
  Creates a rendering of the system. Edit this to
  make it run on matplotlib.

  The Chiral namedtuple has the form [R, theta] where R is an ndarray of
  shape [particle_count, spatial_dimension] while theta is an ndarray of shape
  [particle_count].

  Inputs:
    box_size (float): size-length of box
    states (Chiral namedtuple): special chiral datatype.

  Output:
    anim (Animation): animated rendering of box state, runs at 100fps
  """
  # if states is a TBA: retrieve R and theta
  if isinstance(states, Chiral):
    R = onp.reshape(states.R, (1,) + states.R.shape)
    theta = onp.reshape(states.theta, (1,) + states.theta.shape)

  # if states is a list (sequence of simul. frames)
  elif isinstance(states, list):
    # if all indiv. state recorded in states is a boid: stack all R and theta vectors into an array
    if all([isinstance(x, Chiral) for x in states]):
      R, theta = zip(*states)
      R = onp.stack(R)
      theta = onp.stack(theta)

  # retrieve number of frames
  frames = R.shape[0]

  fig, ax = plt.subplots()

  # formatting plot
  ax.set_xlim(0, box_size)
  ax.set_ylim(0, box_size)

  # single frame rendering
  def renderer_code(frame_num=0):
    """
    Creates an artist list of one simulation frame.
    Only works for 2D.
    """
    if frame_num == frames:
      return []

    # particles data
    curr_R = R[frame_num]
    curr_theta = theta[frame_num]
    curr_x = curr_R[:, 0]
    curr_y = curr_R[:, 1]

    # rendering: USE COLOR TO ENCODE POLARIZATION/ ANGLE OF CHIRALS.
    chiral_plot = ax.scatter(curr_x, curr_y, c=curr_theta, s=.005,
                             cmap="hsv", vmin=0, vmax=2 * onp.pi)
    timer = ax.text(0.5, 1.05, f"t = {DT * DELTA * frame_num:.2f}",
                    size=plt.rcParams["axes.titlesize"],
                    ha="center", transform=ax.transAxes)
    # scatter_plot = [chiral_plot]

    # render_rest = renderer_code(frame_num + 1)
    # render_rest.insert(0, scatter_plot)

    return chiral_plot, timer

  artists = []
  for frame in range(frames):
    artists.append(renderer_code(frame))
  # artists = [renderer_code(frame) for frame in range(frames)]

  # COLORBAR FOR ANGLE
  fig.colorbar(artists[0][0])

  # build the animation
  anim = ani.ArtistAnimation(fig, artists,
                            interval=10, repeat_delay=1000, blit=False)

  plt.close(fig)            # keep the static PNG from appearing
  anim.save(f"{name}.mp4", writer="ffmpeg", dpi=150)
  # return anim
  # display step not done as it is memory-consuming, only used during initial debugging.

Chiral = namedtuple('Chiral', ['R', 'theta'])

# INITIALIZATION

# Create RNG state to draw random numbers (see LINK).
rng = random.PRNGKey(0)

# Periodic boundary conditions:
displacement, shift = space.periodic(box_size)

# Initialize particles
rng, v_rng, omega_rng, R_rng, theta_rng = random.split(rng, 5)

# Initialize Chiral
chiral = Chiral(
    R = box_size * random.uniform(R_rng, (N, 2)),
    theta = random.uniform(theta_rng, (N,), maxval=2*onp.pi)
)

# DYNAMICS

@vmap
def normal(v, theta):
  return np.array([v * np.cos(theta), v * np.sin(theta)])

@vmap
def angle_sum(theta1, theta2):
   init_theta = theta1 + theta2
   return np.where(init_theta < 0, init_theta + 2 * onp.pi,
                   np.where(init_theta > 2 * onp.pi, init_theta - 2 * onp.pi, init_theta))

def align_fn(dr, theta_i, theta_j):
   align_spd = np.sin(theta_j - theta_i)
   dR = space.distance(dr)
   return np.where(dR < 1., align_spd, 0.)

def align_tot(R, theta):
   # Alignment factor
   align = vmap(vmap(align_fn, (0, None, 0)), (0, 0, None))

   # Displacement between all points
   dR = space.map_product(displacement)(R, R)

   return np.sum(align(dR, theta, theta), axis=1)

def dynamics(v, omega, dt=DT, g=0.018, D_r=0.009):
    @jit
    def update(_, state):
        R, theta = state['chiral']
        key = state['key']

        # GENERALIZE LTR
        # Forward-Euler scheme
        n = normal(v, theta)
        dR = n * dt
        dtheta = (omega + g * align_tot(R, theta)) * dt


        # Stochastic step:
        key, split = random.split(key)
        stheta = np.sqrt(2 * D_r * dt) * random.normal(split, theta.shape)

        state['chiral'] = Chiral(
           shift(R, dR),
           angle_sum(theta, dtheta + stheta)
        )
        state['key'] = key

        return state

    return update

def init(mu_v=1, sigma_v=0.4, mu_r=2.2, sigma_r=1.7, N=N):
   # sample from gaussian distrib + filter for positive vals
   spd = mu_v + sigma_v * random.normal(v_rng, (10 * N, ))
   r = mu_r + sigma_r * random.normal(omega_rng, (10 * N, ))

   condition1 = np.where(spd > 0)[0]
   spd = spd[condition1]
   r = r[condition1]

   condition2 = np.where(r > 0)[0]
   spd = spd[condition2]
   r = r[condition2]
   omega = np.divide(spd, r)

   condition3 = np.where(omega <= 1.4)[0]
   spd = spd[condition3]
   omega = omega[condition3]

   spd = spd[:N]
   omega = omega[:N]

   return spd, omega

# SIMULATION

v, omega = init()
update = dynamics(v, omega)

state = {
    'chiral' : chiral,
    'key' : rng,
}

chiral_buffer = []

for i in range(SIMUL_STEPS):
  state = lax.fori_loop(0, DELTA, update, state)
  chiral_buffer += [state['chiral']]

render(box_size, chiral_buffer, f"supekar_{N}_{SIMUL_STEPS}")
