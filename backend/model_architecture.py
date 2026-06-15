import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    GATConv,
    global_mean_pool
)


class TheLearningLoopModel(nn.Module):

    def __init__(
        self,
        node_features=1,
        fp_dim=2048,
        hidden_dim=128,
        output_dim=2,
        gat_dropout=0.3,
        fc_dropout=0.5
    ):
        super().__init__()

        self.conv1 = GATConv(
            node_features,
            hidden_dim,
            heads=4
        )

        self.conv2 = GATConv(
            hidden_dim * 4,
            hidden_dim,
            heads=1
        )

        self.gat_dropout = nn.Dropout(
            p=gat_dropout
        )

        self.fc = nn.Sequential(
            nn.Linear(
                hidden_dim + fp_dim,
                128
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(fc_dropout),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(fc_dropout),

            nn.Linear(64, output_dim)
        )

    def forward(self, data):

        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        fp = data.fp

        x = F.relu(
            self.conv1(x, edge_index)
        )

        x = self.gat_dropout(x)

        x = self.conv2(
            x,
            edge_index
        )

        x = self.gat_dropout(x)

        x = global_mean_pool(
            x,
            batch
        )

        fp = fp.view(
            x.size(0),
            -1
        )

        combined = torch.cat(
            [x, fp],
            dim=1
        )

        return self.fc(combined)