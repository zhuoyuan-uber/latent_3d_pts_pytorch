import torch

x1 = torch.tensor([1.])
x1.requires_grad = True
x2 = torch.tensor([2.])
x2.requires_grad = True
y = x1 ** 2
z = (2 * y).detach()
w = z ** 3 + x2
w.backward()
print('x1', x1.grad)
print('y', y.grad)
print('x2', x2.grad)
