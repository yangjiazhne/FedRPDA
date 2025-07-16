import torch
import random
from torch_geometric.data import Data

def idx_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask

def construct_subgraph_dict_from_node_dict(num_clients, node_dict, G, graph_nx, train, val, test):
    subgraph_list = []
    print(G.y.size(0))
    for client_id in range(num_clients):
        # 创建本地节点索引列表 并且经过随机打乱
        num_local_nodes = len(node_dict[client_id])
        local_node_idx = [idx for idx in range(num_local_nodes)]
        random.shuffle(local_node_idx)

        # 分别计算train val test子集的大小
        train_size = int(num_local_nodes * train)
        val_size = int(num_local_nodes * val)
        test_size = int(num_local_nodes * test)

        # 从shuffle的索引列表进行划分
        train_idx = local_node_idx[: train_size]
        val_idx = local_node_idx[train_size: train_size + val_size]
        test_idx = local_node_idx[train_size + val_size:]

        # 将索引转换为掩码
        local_train_idx = idx_to_mask(train_idx, size=num_local_nodes)
        local_val_idx = idx_to_mask(val_idx, size=num_local_nodes)
        local_test_idx = idx_to_mask(test_idx, size=num_local_nodes)

        map_train_idx = []
        map_val_idx = []
        map_test_idx = []

        # 从 node_dict[client_id] 中获取相应的节点索引，并将它们添加到对应的列表中
        map_train_idx += [node_dict[client_id][idx] for idx in train_idx]
        map_val_idx += [node_dict[client_id][idx] for idx in val_idx]
        map_test_idx += [node_dict[client_id][idx] for idx in test_idx]

        # 将这些列表中的节点索引转换为对应的全局索引掩码
        global_train_idx = idx_to_mask(map_train_idx, size=G.y.size(0))
        global_val_idx = idx_to_mask(map_val_idx, size=G.y.size(0))
        global_test_idx = idx_to_mask(map_test_idx, size=G.y.size(0))

        node_idx_map = {}
        edge_idx = []
        for idx in range(num_local_nodes):
            node_idx_map[node_dict[client_id][idx]] = idx
        edge_idx += [(node_idx_map[x[0]], node_idx_map[x[1]]) for x in
                     graph_nx.subgraph(node_dict[client_id]).edges]
        edge_idx += [(node_idx_map[x[1]], node_idx_map[x[0]]) for x in
                     graph_nx.subgraph(node_dict[client_id]).edges]
        edge_idx_tensor = torch.tensor(edge_idx, dtype=torch.long).T
        subgraph = Data(x=G.x[node_dict[client_id]],
                        y=G.y[node_dict[client_id]],
                        edge_index=edge_idx_tensor)

        subgraph.train_mask = local_train_idx
        subgraph.val_mask = local_val_idx
        subgraph.test_mask = local_test_idx
        subgraph.global_train_mask = global_train_idx
        subgraph.global_val_mask = global_val_idx
        subgraph.global_test_mask = global_test_idx
        subgraph_list.append(subgraph)
        print("Client: {}\tTotal Nodes: {}\tTotal Edges: {}\tTrain Nodes: {}\tVal Nodes: {}\tTest Nodes\t{}".format(
            client_id + 1, subgraph.num_nodes, subgraph.num_edges, train_size, val_size, test_size))
    return subgraph_list

