import math
import numpy as np
from graphviz import Digraph

class Value:
    """Stores a single scalar Value and its gradient"""
    def __init__(self, data, child=(), op=''):
        self.data = data
        self.grad = 0
        self._prev = set(child)
        self._op = op
        self._backward = lambda: None
    
    def __repr__ (self):
        return f"Value(data={self.data}, grad={self.grad})"
    
    def __add__ (self, other):
        other = other if isinstance(other, Value) else Value(other)
        res = Value (self.data + other.data, (self, other), "+")
        def _backward ():
            self.grad += res.grad
            other.grad += res.grad
        res._backward = _backward
        return res

    def __radd__(self, other):
        return self + other
    
    def __mul__ (self, other):
        other = other if isinstance(other, Value) else Value(other)
        res = Value (self.data * other.data, (self, other), "*")
        def _backward ():
            self.grad += res.grad * other.data
            other.grad += res.grad * self.data
        res._backward = _backward
        return res
    
    def __rmull__ (self, other):
        return self * other

    def __pow__ (self, other):
        assert isinstance(other, (float, int)), "Only floating point and integers Values are supported for now"
        res = Value (self.data ** other, (self,), f"**{other}")
        def _backward ():
            self.grad += (other * self.data ** (other - 1)) * res.grad
        res._backward = _backward
        return res
    
    def __neg__ (self):
        return self * -1

    def __sub__ (self, other):
        return self + (-other)
    
    def __rsub__ (self, other):
        return other - self

    def __truediv__ (self, other):
        return self * (other**-1)
    
    def __rtruediv__ (self, other):
        return other / self
    
    def relu(self):
        res = Value(0 if self.data < 0 else self.data, (self,), "ReLU")
        def _backward():
            self.grad += (res.data > 0) * res.grad
        res._backward = _backward
        return res
    
    def _topological_sort(self):
        visited = set()
        sorted = []
        if self not in visited:
            visited.add(self)
            for child in self._prev:
                sorted += child._topological_sort()
            sorted.append(self)
        return sorted

    def backward(self):
        topo = self._topological_sort()
        self.grad = 1
        for v in reversed(topo):
            v._backward()
