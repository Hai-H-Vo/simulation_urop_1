import jax
import equinox as eqx
import numpy as onp
import jax.numpy as np
from jax import random
from jax import jit
from jax import vmap
from jax import lax
from jax import config
config.update("jax_enable_x64", True)

class LegendrePolynomial(eqx.Module):
    params: jax.Array  # shape (max_degree+1,)
    max_degree: int

    def __init__(self, params: jax.Array):
        super().__init__()
        self.params = params
        self.max_degree = len(params) - 1

    def __call__(self, inputs):
        # Inputs are assumed to be in [-1, 1]
        result = self.params[0] * np.ones_like(inputs)
        if self.max_degree >= 1:
            result += self.params[1] * inputs
        p_prev = np.ones_like(inputs)
        p_curr = inputs
        for n in range(2, self.max_degree + 1):
            p_next = ((2 * n - 1) * inputs * p_curr - (n - 1) * p_prev) / n
            result += self.params[n] * p_next
            p_prev, p_curr = p_curr, p_next
        return result

class LaguerrePolynomial(eqx.Module):
    params: jax.Array  # shape (max_degree+1,)
    max_degree: int

    def __init__(self, params: jax.Array):
        super().__init__()
        self.params = params
        self.max_degree = len(params) - 1

    def __call__(self, inputs):
        # Inputs in [0, inf]
        result = self.params[0] * np.ones_like(inputs)
        if self.max_degree >= 1:
            result += np.ones_like(inputs) - self.params[1] * inputs
        p_prev = np.ones_like(inputs)
        p_curr = inputs
        for n in range(2, self.max_degree + 1):
            p_next = (((2 * n - 1) * np.ones_like(inputs) - inputs) * p_curr - (n - 1) * p_prev) / n
            result += self.params[n] * p_next
            p_prev, p_curr = p_curr, p_next
        return result
