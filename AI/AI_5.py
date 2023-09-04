import torch.nn as nn

model = nn.Linear(in_features=1, out_features=1, bias=True)
print(model.weight)
print(model.bias)

class MLP(nn.Module):
    def __init__(self, inputs):
        super(MLP, self).__init__()
        self.layer = nn.Linear(inputs, 1)
        self.activation = nn.Sigmoid()

    def forward(self, X):
        X = self.layer(X)
        X = self.activation(X)
        return X
str().
mlp = MLP(1)
print(mlp.layer)
# mlp.forward(1)