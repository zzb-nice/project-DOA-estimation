import torch
import torch.nn as nn


class multiLayer_model(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(multiLayer_model, self).__init__()
        self.layer = nn.Sequential(*[
            nn.Linear(in_dims, in_dims * 4),
            nn.ReLU(),
            # nn.Linear(in_dims * 2, in_dims * 2),
            # nn.ReLU(),
            # nn.Linear(in_dims * 2, in_dims * 2),
            # nn.ReLU(),
            # nn.Linear(in_dims * 2, in_dims),
            # nn.ReLU(),
            nn.Linear(in_dims*4, out_dims)
        ])

    def forward(self, x):
        x = self.layer(x)
        return x


class multiLayer_model_2(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(multiLayer_model_2, self).__init__()
        self.layer = nn.Sequential(*[
            nn.Linear(in_dims, in_dims*2),
            nn.ReLU(),
            nn.Linear(in_dims*2, in_dims*2),
            nn.ReLU(),
            nn.Linear(in_dims*2, in_dims*4),
            nn.ReLU(),
            nn.Linear(in_dims*4, in_dims*4),
            nn.ReLU(),
            nn.Linear(in_dims*4, in_dims*2),
            nn.ReLU(),
            nn.Linear(in_dims*2, in_dims),
            nn.ReLU(),
            nn.Linear(in_dims, out_dims),
        ])

    def forward(self, x):
        x = self.layer(x)
        return x


class Linear_model(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(Linear_model, self).__init__()
        self.linear = nn.Linear(in_dims, out_dims)

    def forward(self, x):
        x = self.linear(x)

        return x
