import copy
from model import init_model
import torch
from utils.Myutils import test_batch
from torch_geometric.loader import NeighborLoader
import copy
import numpy as np

class Server(object):
    def __init__(self, args, dataset, device):
        self.args = args
        self.model_name = args.model
        self.graph = dataset
        self.input_dim = dataset.num_node_features
        self.hidden_dim = args.hidden_dim
        self.output_dim = args.class_num
        self.model = init_model(self.model_name, self.input_dim, self.output_dim, self.hidden_dim)
        self.round = 0
        self.device = device
        self.graph.to(self.device)
        self.global_pos_prop = None
        self.global_neg_prop = None
        self.a = args.a
        self.b = args.b

        if(self.args.train_batch):
            kwargs = {'batch_size': self.args.batch_size, 'num_workers': 1, 'persistent_workers': True}

            subgraph_loader = NeighborLoader(copy.copy(self.graph), input_nodes=None,
                                             num_neighbors=[-1], shuffle=False, **kwargs)
            del subgraph_loader.data.x, subgraph_loader.data.y
            subgraph_loader.data.num_nodes = self.graph.num_nodes
            subgraph_loader.data.n_id = torch.arange(self.graph.num_nodes)

            self.subgraph_loader = subgraph_loader

    def state_dict(self):
        return self.model.state_dict()
        
    def avg(self, client_list, selected_clients):
        w_list = []
        weights_list1 = []
        
        model = init_model(self.model_name, self.input_dim, self.output_dim, self.hidden_dim)
        model_state = model.state_dict()
        
        global_pos = global_neg = torch.zeros_like(client_list[0].local_pos_prop)

        for i in selected_clients:
            global_pos = global_pos + client_list[i].local_pos_prop / len(selected_clients) 
            global_neg = global_neg + client_list[i].local_neg_prop / len(selected_clients)
            w_list.append(client_list[i].model.state_dict())
            
        for i in selected_clients:
            weights_list1.append(client_list[i].heterogeneity)

        weights_tensor1 = torch.tensor(weights_list1, dtype=torch.float32)

        if weights_tensor1.sum() != 0:
            weights_tensor1 = weights_tensor1 / weights_tensor1.sum()

        if (self.args.dataset == 'tfinance'):
            decay_rate = 5
            round_adjusted = self.round / (self.args.rounds - 1)
            factor = self.a * np.exp(-decay_rate * round_adjusted) + self.b
            
            results = 1 / (1 + np.exp(-factor * (weights_tensor1.numpy() - 0.5)))

        weights_list = results / np.sum(results)
        
        for i in range(len(selected_clients)):
            for key in w_list[i]:
                if i == 0:
                    model_state[key] = w_list[i][key] * weights_list[i]
                else:
                    model_state[key] = model_state[key] + w_list[i][key] * weights_list[i]
        
        for i in range(self.args.clients_nums):
            client_list[i].neg_prop = global_neg.clone()
            client_list[i].pos_prop = global_pos.clone()

        self.global_neg_prop = global_neg.clone()
        self.global_pos_prop = global_pos.clone()

        self.model.load_state_dict(model_state)
        self.round = self.round + 1

    def test_global(self, client_list):
        self.model.to(self.device)
        self.graph.to(self.device)
        # 评估每一个客户端
        client_loss = client_acc = client_f1 = client_precision = client_recall = client_auc = client_aps = 0.0
        for x in range(len(client_list)):
            client_list[x].graph.to(self.device)
            if (self.args.train_batch):
                test_loss, val_acc, test_acc, f1, precision, recall, auc, aps, cm = test_batch(self.model, client_list[x].graph,
                                                                              client_list[x].subgraph_loader, self.device)
                client_loss += test_loss
                client_acc += test_acc
                client_f1 += f1
                client_precision += precision
                client_recall += recall
                client_auc += auc
                client_aps += aps

            print('client{} test loss: {:.4f}'.format(x, test_loss),
                  'client{} Val Accuracy: {:.4f}'.format(x, val_acc),
                  'client{} Test Accuracy: {:.4f}'.format(x, test_acc),
                  'client{} f1: {:.4f}'.format(x, f1),
                  'client{} precision: {:.4f}'.format(x, precision),
                  'client{} recall: {:.4f}'.format(x, recall),
                  'client{} auc: {:.4f}'.format(x, auc),
                  'client{} aps: {:.4f}'.format(x, aps),
                  'client{} cm: ', cm)
        
        client_loss = client_loss / len(client_list)
        client_acc = client_acc / len(client_list)
        client_f1 = client_f1 / len(client_list)
        client_precision = client_precision / len(client_list)
        client_recall = client_recall / len(client_list)
        client_auc = client_auc / len(client_list)
        client_aps = client_aps / len(client_list)

        print('CLient average Loss: {:.4f}'.format(client_loss),
            'CLient average Accuracy: {:.4f}'.format(client_acc),
            'CLient average f1: {:.4f}'.format(client_f1),
            'CLient average precision: {:.4f}'.format(client_precision),
            'CLient average recall: {:.4f}'.format(client_recall),
            'CLient average auc: {:.4f}'.format(client_auc),
            'CLient average aps: {:.4f}'.format(client_aps))

        if (self.args.train_batch):
            test_loss, val_acc, test_acc, f1, precision, recall, auc, aps, cm = test_batch(self.model, self.graph,
                                                                          self.subgraph_loader, self.device)

        print('Server test loss: {:.4f}'.format(test_loss),
              'Server Accuracy: {:.4f}'.format(test_acc),
              'Server f1: {:.4f}'.format(f1),
              'Server precision: {:.4f}'.format(precision),
              'Server recall: {:.4f}'.format(recall),
              'Server auc: {:.4f}'.format(auc),
              'Server aps: {:.4f}'.format(aps),
              'Server cm: ', cm)
        print('----------------------------------------------------')