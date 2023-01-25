import random
from engine import value

class Module:
    
    def zero_grad (self):
        for p in self.parameters():
            p.grad = 0
    
    def parameters (self):
        return []

class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        self.weights = [value(random.uniform(-1, 1)) for _ in range(nin)]
        self.base = value(random.uniform(-5, 5))
        self.nonlin = nonlin
    
    def __call__ (self, x):
        act = sum((wi * xi for wi,xi in zip(self.weights, x)), self.base)
        return act.relu() if self.nonlin else act
    
    def parameters (self):
        return self.weights + [self.base]
    
    def __repr__ (self):
        return f"{'ReLU' if self.nonlin else 'Linear'} neuron ({len(self.weights)})"

class Layer(Module):
    
    def __init__ (self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range (nout)]
    
    def __call__ (self, x):
        res = [n(x) for n in self.neurons]
        return res[0] if len(res) == 1 else res
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
    
    def __repr__ (self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"
    
class MLP(Module):

    def __init__(self, nin, nout):
        sz = [nin] + nout
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nout)-1) for i in range (len(nout))]

    def __call__ (self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
