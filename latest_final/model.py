from __future__ import division
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from utils import norm_col_init, weights_init


class A3Clstm(torch.nn.Module):
    def __init__(self, num_inputs, action_space):
        super(A3Clstm, self).__init__()

        self.basic = nn.Sequential(
            nn.Conv2d(num_inputs, 32, 5, stride=1, padding=2),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 5, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU()
        )

        self.lstm = nn.LSTMCell(1024, 512)
        num_outputs = action_space.n
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, num_outputs)

        self.apply(weights_init)
        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()

    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        x = self.basic(inputs)
        x = x.view(x.size(0), -1)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx
        return self.critic_linear(x), self.actor_linear(x), (hx, cx)
