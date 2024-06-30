import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvBlock(nn.Module):

    def __init__(self, input_channel, output_channel):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channel))

    def forward(self, inp):
        return self.layers(inp)


class BackBone(nn.Module):

    def __init__(self, num_channel=64):
        super().__init__()

        self.ada_pool = nn.AdaptiveAvgPool2d(1)
        self.layer1 = nn.Sequential(
            ConvBlock(3, num_channel),
            nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.ca1 = nn.Sequential(
            nn.Linear(num_channel, num_channel, bias=False),
            nn.Sigmoid())
        self.pool1 = nn.MaxPool2d((8, 8))

        self.layer2 = nn.Sequential(
            ConvBlock(num_channel, num_channel),
            nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.ca2 = nn.Sequential(
            nn.Linear(num_channel, num_channel, bias=False),
            nn.Sigmoid())
        self.pool2 = nn.MaxPool2d((4, 4))

        self.layer3 = nn.Sequential(
            ConvBlock(num_channel, num_channel),
            nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.ca3 = nn.Sequential(
            nn.Linear(num_channel, num_channel, bias=False),
            nn.Sigmoid())
        self.pool3 = nn.MaxPool2d((2, 2))

        self.layer4 = nn.Sequential(
            ConvBlock(num_channel, num_channel),
            nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.ca4 = nn.Sequential(
            nn.Linear(num_channel, num_channel, bias=False),
            nn.Sigmoid())

        self.fusion = nn.Sequential(
            nn.Conv2d(256, num_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channel), nn.ReLU(inplace=True))

    def forward(self, inp):
        x1 = self.layer1(inp)

        a1 = self.ada_pool(x1)
        a1 = a1.squeeze(-1).squeeze(-1)
        c1 = self.ca1(a1)
        c1 = c1.unsqueeze(-1).unsqueeze(-1)
        p1 = c1 * self.pool1(x1)

        x2 = self.layer2(x1)
        a2 = self.ada_pool(x2)
        a2 = a2.squeeze(-1).squeeze(-1)
        c2 = self.ca2(a2)
        c2 = c2.unsqueeze(-1).unsqueeze(-1)
        p2 = c2 * self.pool2(x2)

        x3 = self.layer3(x2)
        a3 = self.ada_pool(x3)
        a3 = a3.squeeze(-1).squeeze(-1)
        c3 = self.ca3(a3)
        c3 = c3.unsqueeze(-1).unsqueeze(-1)
        p3 = c3 * self.pool3(x3)

        x4 = self.layer4(x3)
        a4 = self.ada_pool(x4)
        a4 = a4.squeeze(-1).squeeze(-1)
        c4 = self.ca4(a4)
        c4 = c4.unsqueeze(-1).unsqueeze(-1)
        p4 = c4 * x4

        cat_x = torch.cat([p1, p2, p3, p4], dim=1)
        out = self.fusion(cat_x)

        return out
