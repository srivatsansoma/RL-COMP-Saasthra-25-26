import torch
import torch

print(torch.cuda.is_available())

x = torch.ones(5, device="cuda")  # input tensor
y = torch.zeros(3, device="cuda")  # expected output
w = torch.randn(5, 3, requires_grad=True, device="cuda")
b = torch.randn(3, requires_grad=True, device="cuda")
z = torch.matmul(x, w)+b

loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

print(w.grad)
print(b.grad)

loss.backward()

print(w.grad)
print(b.grad)