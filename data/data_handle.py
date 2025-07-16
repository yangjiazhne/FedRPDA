from sklearn.preprocessing import StandardScaler
import torch
from dgl.data.utils import load_graphs
from torch_geometric.data import Data

def load_tfinance():
    print('====================================================================')
    print('tfinance...')
    RAW_PATH = '/nfs5/yjz/dataset/raw/'

    # 加载图数据
    g, _ = load_graphs(RAW_PATH + 'tfinance')
    g = g[0]

    # 特征标准化
    X = g.ndata['feature'].numpy()
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    x = torch.FloatTensor(X_std)

    # 提取标签
    y = g.ndata['label'][:, 1].numpy()
    y = torch.LongTensor(y)

    src_nodes = g.edges()[0]
    dst_nodes = g.edges()[1]

    edge_index = torch.stack([src_nodes, dst_nodes], dim=0)

    MyGraph = Data(x=x, y=y, edge_index=edge_index)

    return MyGraph

def load_data(data_name):
    graph = None
    if data_name == 'tfinance':
        graph = load_tfinance()
    return graph