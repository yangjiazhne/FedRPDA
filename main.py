import argparse
from trainer.Client import Client
from trainer.Server import Server
from trainer.Trainer import Trainer
from data.data_handle import load_data
import torch
import os
import sys
from utils.Myutils import Logger
import os.path as osp
import time

parser = argparse.ArgumentParser(description='Insert Arguments')

parser.add_argument("--dataset", type=str, default="tfinance", help="dataset used for training")
parser.add_argument("--clients_nums", type=int, default=4, help="number of clients")
parser.add_argument('--client_sample_ratio', type=float, default=1, help='client sample ratio')
parser.add_argument("--hidden_dim", type=int, default=64, help="size of GNN hidden layer")
parser.add_argument("--class_num", type=int, default=2, help="class num")
parser.add_argument("--epochs", type=int, default=5, help="epochs for training")
parser.add_argument("--rounds", type=int, default=200, help="federated rounds performed")
parser.add_argument('--model', type=str, default="sage_batch", help='gnn model')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate of gnn model')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay of gnn model')
parser.add_argument('--dropout', type=float, default=0.0, help='drop out of gnn model')
parser.add_argument('--train_batch', type=bool, default=True, help='whether train batch')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--lamda', type=int, default=2, help='lamda')
parser.add_argument('--a', type=int, default=7, help='a')
parser.add_argument('--b', type=int, default=4, help='b')
parser.add_argument('--save_dir', default='./outputs/log', type=str, help='save directory')

args = parser.parse_args()

loca=time.strftime('%Y-%m-%d-%H-%M-%S)')
sys.stdout = Logger(osp.join(args.save_dir, args.dataset + loca + '.txt'))  # 记录训练过程中的输出日志

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

def get(idx):
    data = torch.load(osp.join('/nfs5/yjz/dataset/FLG_data/processed/{}'.format(args.dataset), 'client{}_subgraph_{}'.format(idx, args.dataset)))
    return data

# 加载数据 && 划分数据
global_graph = load_data(args.dataset)
client_subgraph_list = [get(i) for i in range(args.clients_nums)]

for i in range(len(client_subgraph_list)):
    if i == 0:
        global_graph.train_mask = client_subgraph_list[i].global_train_mask
        global_graph.val_mask = client_subgraph_list[i].global_val_mask
        global_graph.test_mask = client_subgraph_list[i].global_test_mask
    else:
        global_graph.train_mask += client_subgraph_list[i].global_train_mask
        global_graph.val_mask += client_subgraph_list[i].global_val_mask
        global_graph.test_mask += client_subgraph_list[i].global_test_mask

client_list = []
for z in range(args.clients_nums):
    Cl = Client(args, client_subgraph_list[z], z, device)
    client_list.append(Cl)

# 创建服务端
server = Server(args=args, dataset=global_graph, device=device)

trainer = Trainer(
    server=server,
    clients=client_list,
    args=args
)
trainer.train()
