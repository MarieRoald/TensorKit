import numpy as np
import base
import cp

class Logger:

    def __init__(self, args, loss, grad):
        self.loss_values = []
        self.gradients = []
        self.gradient_values = []   
        self.args = args

        self.loss = loss
        self.grad = grad


    def log(self, parameters):

        loss = self.loss(parameters, *self.args)

        grad = self.grad(parameters, *self.args)
        gradients = base.unflatten_factors(grad, *self.args[:2])

        self.loss_values.append(loss)
        self.gradients.append(gradients)
        self.gradient_values.append(np.linalg.norm(grad))

        