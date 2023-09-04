import torch

a = torch.tensor([[1, 2, 3, 4], [3, 4, 5, 6]], dtype=torch.int8)
b = torch.tensor([[1, 2, 3, 4], [3, 4, 5, 6]], dtype=torch.int8) + 3
print(a.cpu().numpy())
print(a[1][0])
print(a[1, 0])
print(a[0][1:3])
print(b)
print(a.shape)
print(a+b)
c= a+b
print(c.view(8,1))
print(c.view(1,8))
print(c.view(2,-1,2).shape)
print(torch.cuda.is_available())