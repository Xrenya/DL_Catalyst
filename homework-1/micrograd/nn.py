import numpy as np

from engine import Value, Tensor


class Module:
    """
    Base class for every layer.
    """
    def forward(self, *args, **kwargs):
        """Depends on functionality"""
        pass

    def __call__(self, *args, **kwargs):
        """For convenience we can use model(inp) to call forward pass"""
        return self.forward(*args, **kwargs)

    def parameters(self):
        """Return list of trainable parameters"""
        return []


class Linear(Module):
    def __init__(self, in_features, out_features, bias: bool = True):
        """Initializing model"""
        # Create Linear Module
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight = WeightInitializer(self.out_features, self.in_features, self.bias).initialize()[0]
        self.bias = WeightInitializer(self.out_features, self.in_features, self.bias).initialize()[1]

    def forward(self, input):
        """Y = W * x + b"""
        output = np.matmul(input, self.weight.T)
        if self.bias is True:
            output += self.bias
        return output

    def parameters(self):
        return self.weight, self.bias

class WeightInitializer:
    """Initializing weights"""
    def __init__(self, out_features, in_features, bias: bool = True):
        self.out_features = out_features
        self.in_features = in_features
        self.bias = bias
        
    def initialize(self):
        std = np.sqrt(2 / self.in_features)
        weight = np.random.normal(loc=0, scale=std, size=(self.out_features, self.in_features))
        bias = np.random.normal(loc=0, scale=std, size=self.out_features) if self.bias is True else 0
        return weight, bias

class ReLU(Module):
    """The most simple and popular activation function"""
    def forward(self, inp):
        # Create ReLU Module
        return Tensor(inp).relu()


class CrossEntropyLoss(Module):
    """Cross-entropy loss for multi-class classification"""
    def forward(self, inp, label):
        # Create CrossEntropy Loss Module
        return -np.sum(label * np.log(inp), axis=1)
