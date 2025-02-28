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
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads, edge_feature_dim, dropout_rate=0.1):
        super(GATModel, self).__init__()
        self.dropout_rate = dropout_rate  # Dropout rate

        # Define the GAT layers
        self.gat1 = GATv2Conv(in_channels, hidden_channels, heads=num_heads, edge_dim=edge_feature_dim)
        self.gat2 = GATv2Conv(hidden_channels * num_heads, hidden_channels, heads=num_heads, edge_dim=edge_feature_dim)
        self.gat3 = GATv2Conv(hidden_channels * num_heads, hidden_channels, heads=num_heads, edge_dim=edge_feature_dim)
        self.gat4 = GATv2Conv(hidden_channels * num_heads, hidden_channels, heads=num_heads, edge_dim=edge_feature_dim)
        self.gat5 = GATv2Conv(hidden_channels * num_heads, out_channels, heads=num_heads, concat=False, edge_dim=edge_feature_dim)

        # Batch Normalization layers
        self.bn1 = nn.BatchNorm1d(hidden_channels * num_heads)
        self.bn2 = nn.BatchNorm1d(hidden_channels * num_heads)
        self.bn3 = nn.BatchNorm1d(hidden_channels * num_heads)
        self.bn4 = nn.BatchNorm1d(hidden_channels * num_heads)

        # Dropout layer
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, x, edge_index, edge_attr):
        # Pass through the GAT layers with BatchNorm, Dropout, and activation
        x = self.gat1(x, edge_index, edge_attr)
        x = self.bn1(x)  # Batch Normalization
        x = F.elu(x)  # Activation function
        x = self.dropout(x)  # Dropout after activation

        x = self.gat2(x, edge_index, edge_attr)
        x = self.bn2(x)  # Batch Normalization
        x = F.elu(x)
        x = self.dropout(x)

        x = self.gat3(x, edge_index, edge_attr)
        x = self.bn3(x)  # Batch Normalization
        x = F.elu(x)
        x = self.dropout(x)

        x = self.gat4(x, edge_index, edge_attr)
        x = self.bn4(x)  # Batch Normalization
        x = F.elu(x)
        x = self.dropout(x)

        x, attn_weights_5 = self.gat5(x, edge_index, edge_attr, return_attention_weights=True)

        # Compute edge outputs using node embeddings
        edge_outputs = torch.sum(x[edge_index[0]] * x[edge_index[1]], dim=-1)

        # Return edge outputs and attention weights from all layers
        return edge_outputs, attn_weights_5

    def predict(self, x, edge_index, edge_attr):
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient computation for inference
            edge_outputs, attn_weights_5 = self.forward(x, edge_index, edge_attr)
            return edge_outputs, attn_weights_5
