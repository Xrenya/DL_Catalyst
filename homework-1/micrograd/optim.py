class SGD:
    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        """Applying gradient descent to parameters"""
        # Implement SGD!
        for i, parameter in enumerate(self.parameters):
            self.parameters[i] -= self.lr * self.parameters[i]

    def zero_grad(self):
        """Resetting gradient for all parameters (set gradient to zero)"""
        for i, parameter in enumerate(self.parameters):
            self.parameters[i].grad = 0