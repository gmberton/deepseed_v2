
import os
import torch
import argparse
import torch.nn as nn

import models
import commons

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--seed_weights", type=int, default=0, help="_")

args = parser.parse_args()
commons.make_deterministic(args.seed_weights)

net = models.build_network()
net = net.half()
for layer in net.modules():
    if isinstance(layer, nn.BatchNorm2d):
        layer = layer.float()
        if hasattr(layer, 'weight') and layer.weight is not None:
            _ = layer.weight.data.fill_(1.0)
        layer.eps = 0.00001
        layer.momentum = 0.1

os.makedirs("weights", exist_ok=True)
torch.save(net.state_dict(), f"weights/{args.seed_weights:03d}.pth")

