import torch.nn as nn
import torch.nn.functional as F

class SRCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.c1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)

        self.c2 = nn.Conv2d(64, 32, kernel_size=1)

        self.c3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        return self.c3(x)