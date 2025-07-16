import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch.nn import ModuleList

class SAGEBATCH(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden=64, dropout=0.0, num_layers=3):
        super(SAGEBATCH, self).__init__()
        self.convs = ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.convs.append(SAGEConv(in_channels, hidden))
            elif (i + 1) == num_layers:
                self.convs.append(SAGEConv(hidden, out_channels))
            else:
                self.convs.append(SAGEConv(hidden, hidden))
        self.dropout = dropout
        self.num_layers = num_layers

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        features = []
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            features.append(x)
            if (i + 1) == len(self.convs):
                break
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        outputs = {"logits": x,
                   "feature": features[-2]}

        return outputs

    def inference(self, x_all, subgraph_loader, device):
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id.to(x_all.device)].to(device)
                x = conv(x, batch.edge_index.to(device))
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x[:batch.batch_size])
            x_all = torch.cat(xs, dim=0)
        return x_all