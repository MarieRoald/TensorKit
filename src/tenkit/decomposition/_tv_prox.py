import numpy as np
from scipy.optimize import minimize


def unflatten_vector(x, arrays):
    new_offset = 0
    for array in arrays:
        offset = new_offset
        new_offset = offset + np.prod(array.shape)

        current_view = x[offset:new_offset]
        array[:] = current_view.reshape(array.shape)
    
    return arrays


def flatten_vector(arrays, out=None):
    if out is None:
        shape = sum(np.prod(a.shape) for a in arrays)
        out = np.empty(shape)
    new_offset = 0
    for array in arrays:
        offset = new_offset
        new_offset = offset + np.prod(array.shape)

        out[offset:new_offset] = array.ravel()
    
    return out


class TotalVariation:
    def __init__(self, center, reg_strength):
        self.center = center
        self.reg_strength = reg_strength

        self._reg_vector = np.ones((center.shape[0], 1))*reg_strength
        self._reg_vector[0] = 0
        shape = center.shape

        self._shape = shape

        self.positive = np.maximum(center, 0)
        self.negative = np.maximum(-center, 0)

        self.params = [self.positive, self.negative]

        # _vector_shape is the length of the vectorised factor matrix
        # after splitting it in a positive and negative part.
        self._vector_shape = 2*(shape[0])*shape[1]
    
    @property
    def reg_strength(self):
        return self._reg_strength
    
    @reg_strength.setter
    def reg_strength(self, value):
        self._reg_strength = value

        self._reg_vector = np.ones((self.center.shape[0], 1))*value
        self._reg_vector[0] = 0

    def flatten_params(self, params, out=None):
        if out is None:
            out = np.empty(self._vector_shape)
        
        return flatten_vector(self.params, out)
    
    def unflatten_params(self, x):
        self.params = unflatten_vector(x, self.params)

    def compute_error(self):
        params = self.positive - self.negative
        integrated_params = np.cumsum(params, axis=0)
        error = integrated_params - self.center

        return error

    def compute_sse(self):
        error = self.compute_error()
        return np.sum(error**2)

    def loss(self):
        regulariser = self.reg_strength*(self.positive[1:].sum() + self.negative[1:].sum())
        sse = self.compute_sse()
        return regulariser + 0.5*sse

    def center_penalty(self):
        return self.reg_strength*np.sum(np.abs(self.center[1:] - self.center[:-1]))

    def gradient(self):
        error = self.compute_error()
        integrate_T_error = np.cumsum(error[::-1], axis=0)[::-1]
        return [integrate_T_error + self._reg_vector, -integrate_T_error + self._reg_vector]
    
    def scipy_loss(self, x):
        self.unflatten_params(x)
        return self.loss()
    
    def scipy_grad(self, x):
        self.unflatten_params(x)
        grad = self.gradient()
        return flatten_vector(grad)

    def prox(self):
        res = minimize(
            fun=self.scipy_loss,
            x0=flatten_vector(self.params),
            method='L-BFGS-B',
            jac=self.scipy_grad,
            bounds=[(0, np.inf)]*len(flatten_vector(self.params)),
            tol=1e-3
        )
        self.unflatten_params(res.x)
        return np.cumsum(self.params[0] - self.params[1], axis=0)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from numba import vectorize

    def square_wave(x):
        return float((x % 10) < 5)

    a1 = 20*np.array([square_wave(xi) for xi in np.linspace(0, 50, 320)])
    a2 = 20*np.array([square_wave(xi/2) for xi in np.linspace(0, 50, 320)])
    A = np.stack([a1, a2]).T
    A_noise = A + np.random.standard_normal(A.shape)*2


    tv = TotalVariation(A_noise, 100)
    denoised = tv.prox()
    plt.close()
    plt.plot(A_noise, '.')
    plt.plot(denoised)
    plt.show()