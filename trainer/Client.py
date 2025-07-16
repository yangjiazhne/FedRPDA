from model import init_model
import torch
from utils.Myutils import test_batch, train_batch, get_local_prop
from torch_geometric.loader import NeighborLoader
import copy
import networkx as nx
from torch_geometric.utils import to_networkx

class Client(object):
    def __init__(self, args, dataset, id, device):
        self.args = args
        self.id = id
        self.model_name = args.model
        self.graph = dataset
        self.num_nodes = dataset.num_nodes
        self.input_dim = dataset.num_node_features
        self.hidden_dim = args.hidden_dim
        self.output_dim = args.class_num
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.dropout = args.dropout
        self.epochs = args.epochs
        self.model = init_model(self.model_name, self.input_dim, self.output_dim, self.hidden_dim)
        self.heterogeneity = self.node_heterogeneity(self.graph)
        self.lamda = args.lamda
        self.neg_prop = None
        self.pos_prop = None
        self.local_neg_prop = None
        self.local_pos_prop = None

        total_train_nodes = self.graph.train_mask.sum().item()
        self.pos_node_ratio = torch.sum(self.graph.y[self.graph.train_mask] == 1) / total_train_nodes
        self.neg_node_ratio = torch.sum(self.graph.y[self.graph.train_mask] == 0) / total_train_nodes
        print(total_train_nodes, torch.sum(self.graph.y[self.graph.train_mask] == 1), self.pos_node_ratio, torch.sum(self.graph.y[self.graph.train_mask] == 0), self.neg_node_ratio)

        self.device = device
        self.graph.to(self.device)
        if(self.args.train_batch):
            kwargs = {'batch_size': self.args.batch_size, 'num_workers': 1, 'persistent_workers': True}
            self.train_loader = NeighborLoader(self.graph, input_nodes=self.graph.train_mask,
                                          num_neighbors=[25, 10], shuffle=True, **kwargs)

            subgraph_loader = NeighborLoader(copy.copy(self.graph), input_nodes=None,
                                             num_neighbors=[-1], shuffle=False, **kwargs)

            del subgraph_loader.data.x, subgraph_loader.data.y
            subgraph_loader.data.num_nodes = self.graph.num_nodes
            subgraph_loader.data.n_id = torch.arange(self.graph.num_nodes)

            self.subgraph_loader = subgraph_loader

    def node_heterogeneity(self, data):
        G = to_networkx(data, to_undirected=True)
        pagerank_scores = nx.pagerank(G, alpha=0.85)
        fraud_nodes = (data.y == 1).nonzero(as_tuple=False).view(-1).tolist()
        rho1 = sum(pagerank_scores[i] for i in fraud_nodes)
        
        return rho1

    def update(self, model_state_dict):
        self.model = init_model(self.model_name, self.input_dim, self.output_dim, self.hidden_dim)
        self.model.load_state_dict(state_dict=copy.deepcopy(model_state_dict))

    def train_local_model(self):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = torch.nn.CrossEntropyLoss()
        print(f'Client ID: {self.id:02d} Starting Local Training')
        self.model.to(self.device)
        self.graph.to(self.device)

        self.local_neg_prop = None
        self.local_pos_prop = None

        for epoch in range(1, self.epochs + 1):
            if (self.args.train_batch):
                loss_ce, loss2, acc = train_batch(self.model, self.train_loader, optimizer, criterion, self.device, self.neg_prop, self.pos_prop, self.lamda)
                test_loss, val_acc, test_acc, f1, precision, recall, auc, aps, cm = test_batch(self.model, self.graph, self.subgraph_loader, self.device)

            if (epoch == self.epochs):
                self.local_pos_prop, self.local_neg_prop = get_local_prop(self.model, self.train_loader, self.device)

        print('Client: {}'.format(self.id),
              'loss_ce_train: {:.4f}'.format(loss_ce),
              'loss_2_train: {:.4f}'.format(loss2),
              'loss_test: {:.4f}'.format(test_loss),
              'acc_train: {:.4f}'.format(acc),
              'acc_val: {:.4f}'.format(val_acc),
              'acc_test: {:.4f}'.format(test_acc),
              'f1: {:.4f}'.format(f1),
              'precision: {:.4f}'.format(precision),
              'recall: {:.4f}'.format(recall),
              'auc: {:.4f}'.format(auc),
              'aps: {:.4f}'.format(aps),
              'cm: ', cm)

