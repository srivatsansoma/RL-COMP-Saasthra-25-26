import torch
from torch import nn
import torch.nn.functional as F

class dqn(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim = [32,32]):
        super(dqn, self).__init__()
        self.ll2 = nn.Linear(input_dim, output_dim)
        self.hidden_layers = []
        
        prev_dim = input_dim
        for h_dim in hidden_dim:
            self.hidden_layers.append(nn.Linear(prev_dim, h_dim))
            prev_dim = h_dim
        self.model = nn.Sequential(self.ll2, *self.hidden_layers)

    def forward(self, x):
        return self.model(x)
    
if __name__ == "__main__":
    model = dqn(2,2)
    print(*model.parameters())