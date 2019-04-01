import torch as t
a = t.Tensor([-0.5,0.5])
f = lambda x:1/(1+t.exp(-6*x))
b = f(a)
print(a)
print(b)
