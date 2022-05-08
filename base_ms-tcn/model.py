#!/usr/bin/python2.7
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np
# from loguru import logger

class MS_TCN2(nn.Module):
    def __init__(self, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes):
        super(MS_TCN2, self).__init__()
        self.PG = Prediction_Generation(num_layers_PG, num_f_maps, dim, num_classes)
        self.Rs = nn.ModuleList([copy.deepcopy(Refinement(num_layers_R, num_f_maps, num_classes, num_classes)) for s in range(num_R)])

    def forward(self, x, mask):
        out = self.PG(x, mask)
        outputs = out.unsqueeze(0)
        for R in self.Rs:
            out = R(F.softmax(out, dim=1), mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)),dim=0)
        return outputs

class Prediction_Generation(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(Prediction_Generation, self).__init__()
        self.num_layers = num_layers
        self.conv_in = nn.Conv1d(dim, num_f_maps, 1)
        self.conv_dilated_down = nn.ModuleList(
            (nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2 ** (num_layers-1-i), dilation=2**(num_layers-1-i)) for i in range(num_layers))
        )
        self.conv_dilated_up = nn.ModuleList(
            (nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2 ** i, dilation=2 ** i) for i in range(num_layers))
        )
        self.conv_fusion = nn.ModuleList(
            (nn.Conv1d(2*num_f_maps, num_f_maps, 1) for i in range(num_layers))
        )
        self.dropout = nn.Dropout()
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
    def forward(self, x, mask):
        f = self.conv_in(x)
        for i in range(self.num_layers):
            f_in = f
            f_down = self.conv_dilated_down[i](f)
            f_up = self.conv_dilated_up[i](f)
            f = self.conv_fusion[i](torch.cat([f_down, f_up],1))
            f = F.relu(f)
            f = self.dropout(f)
            f = f + f_in
        out = self.conv_out(f)
        return out

class Refinement(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(Refinement, self).__init__()
        self.conv_in = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2**i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_in(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out)
        return out

class MultiStageModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MultiStageModel, self).__init__()
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(num_layers, num_f_maps, num_classes, num_classes)) for s in range(num_stages-1)])

    def forward(self, x, mask):
        out = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask[:, 0:1, :]
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out)
        # return (x + out) * mask[:, 0:1, :]



