from collections import defaultdict
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import mean_squared_error, mean_absolute_error

import random
class GATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads, edge_feature_dim, dropout_rate=0.2):
        super(GATModel, self).__init__()
        self.dropout_rate = dropout_rate

        # Define the GAT layers
        self.gat1 = GATv2Conv(in_channels, hidden_channels, heads=num_heads, edge_dim=edge_feature_dim)
        self.gat2 = GATv2Conv(hidden_channels * num_heads, hidden_channels, heads=num_heads, edge_dim=edge_feature_dim)
        self.gat3 = GATv2Conv(hidden_channels * num_heads, hidden_channels, heads=num_heads, edge_dim=edge_feature_dim)
        self.gat4 = GATv2Conv(hidden_channels * num_heads, hidden_channels, heads=num_heads, edge_dim=edge_feature_dim)
        self.gat5 = GATv2Conv(hidden_channels * num_heads, out_channels, heads=num_heads, concat=False, edge_dim=edge_feature_dim)

        # LayerNorm layers (replacing BatchNorm)
        self.ln1 = nn.LayerNorm(hidden_channels * num_heads)
        self.ln2 = nn.LayerNorm(hidden_channels * num_heads)
        self.ln3 = nn.LayerNorm(hidden_channels * num_heads)
        self.ln4 = nn.LayerNorm(hidden_channels * num_heads)

        # Dropout layer
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, x, edge_index, edge_attr):
        x = self.gat1(x, edge_index, edge_attr)
        x = self.ln1(x)  # LayerNorm instead of BatchNorm
        x = F.elu(x)
        x = self.dropout(x)

        x = self.gat2(x, edge_index, edge_attr)
        x = self.ln2(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.gat3(x, edge_index, edge_attr)
        x = self.ln3(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.gat4(x, edge_index, edge_attr)
        x = self.ln4(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.gat5(x, edge_index, edge_attr)

        edge_outputs = torch.sum(x[edge_index[0]] * x[edge_index[1]], dim=-1)

        return edge_outputs, x

    def predict(self, x, edge_index, edge_attr):
        self.eval()
        with torch.no_grad():
            edge_outputs, node_embeddings = self.forward(x, edge_index, edge_attr)
            return node_embeddings, edge_outputs
