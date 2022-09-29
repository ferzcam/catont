import torch.nn as nn
import torch.nn.functional as F


class OrderedMorphism(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        pass

    def forward(self, a, b):
        loss = F.relu(a - b)
        return loss


class TransformationMorphism(nn.Module):
    def __init__(self, embedding_size) -> None:
        super().__init__()
        self.fc = nn.Linear(embedding_size, embedding_size)
        self.criterion = nn.MSELoss()

    def forward(self, a):
        return self.fc(a)
