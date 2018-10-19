import numpy as np
import base
import cp

class Logger:

    def __init__(self, rank, sizes, X):
        self.loss_values = []
        self.gradients = []
        self.gradient_values = []        
        self.rank = rank
        self.sizes = sizes
        self.X = X


    def log(self, parameters):

        loss = cp._cp_loss_scipy(parameters, self.rank, self.sizes, self.X)

        grad = cp._cp_grad_scipy(parameters, self.rank, self.sizes, self.X)
        gradients = base.unflatten_factors(grad, self.rank, self.sizes)

        self.loss_values.append(loss)
        self.gradients.append(gradients)
        self.gradient_values.append(np.linalg.norm(grad))

        