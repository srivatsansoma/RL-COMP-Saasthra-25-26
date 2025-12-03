import torch
from torch import nn
import torch.nn.functional as F

class dqn(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim = [32,32]):
        super(dqn, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        layers = []
        for i in range(len(hidden_dim)):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim[i]))
            else:
                layers.append(nn.Linear(hidden_dim[i-1], hidden_dim[i]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim[-1], output_dim))
        self.layers = nn.Sequential(*layers).to(self.device)

    def forward(self, x):
        return self.layers(x)
    
if __name__ == "__main__":
    model = dqn(2,2)
    print(*model.parameters())