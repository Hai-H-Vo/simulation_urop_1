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

from functools import partial

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
            result += self.params[1] * (np.ones_like(inputs) - inputs)
        p_prev = np.ones_like(inputs)
        p_curr = inputs
        for n in range(2, self.max_degree + 1):
            p_next = (((2 * n - 1) * np.ones_like(inputs) - inputs) * p_curr - (n - 1) * p_prev) / n
            result += self.params[n] * p_next
            p_prev, p_curr = p_curr, p_next
        return result

class LegendreBase(eqx.Module):
    degree: int

    def __init__(self, degree: int):
        super().__init__()
        self.degree = degree

    def __call__(self, inputs):
        # Inputs are assumed to be in [-1, 1]
        result = np.ones_like(inputs)

        result = lax.cond(self.degree >= 1,
                          lambda res, inp: res + inp,
                          lambda res, inp: res,
                          result, inputs)

        p_prev = np.ones_like(inputs)
        p_curr = inputs

        init_val = {
            'prev' : p_prev,
            'curr' : p_curr,
            'result' : result,
            'degree' : 2
        }

        def _body_fun(_, val, inp):
            p_prev = val['prev']
            p_curr = val['curr']
            n = val['degree']
            result = val['result']

            p_next = ((2 * n - 1) * inp * p_curr - (n - 1) * p_prev) / n
            result += p_next
            return {
                'prev' : p_curr,
                'curr' : p_next,
                'result' : result,
                'degree' : n + 1
            }

        body_fun = partial(_body_fun, inp=inputs)

        final_val = lax.fori_loop(2, self.degree + 1, body_fun, init_val)

        return final_val['result']


class LaguerreBase(eqx.Module):
    degree: int

    def __init__(self, degree: int):
        super().__init__()
        self.degree = degree

    def __call__(self, inputs):
        # Inputs in [0, inf]
        result = np.ones_like(inputs)

        result = lax.cond(self.degree >= 1,
                    lambda res, inp: res + np.ones_like(inp) - inp,
                    lambda res, inp: res,
                    result, inputs)

        p_prev = np.ones_like(inputs)
        p_curr = inputs

        init_val = {
            'prev' : p_prev,
            'curr' : p_curr,
            'result' : result,
            'degree' : 2
        }

        def _body_fun(_, val, inp):
            p_prev = val['prev']
            p_curr = val['curr']
            n = val['degree']
            result = val['result']

            p_next = (((2 * n - 1) * np.ones_like(inp) - inp) * p_curr - (n - 1) * p_prev) / n
            result += p_next
            return {
                'prev' : p_curr,
                'curr' : p_next,
                'result' : result,
                'degree' : n + 1
            }

        body_fun = partial(_body_fun, inp=inputs)

        final_val = lax.fori_loop(2, self.degree + 1, body_fun, init_val)

        return final_val['result']
