from torch_geometric.utils import to_networkx
from louvain.community import community_louvain
from data.data_utils import construct_subgraph_dict_from_node_dict

def LouvainSplitter(G, client_num, train_rate = 0.4, val_rate = 0.2, test_rate = 0.2):
    graph_nx = to_networkx(G, to_undirected=True)

    partition = community_louvain.best_partition(graph_nx)

    groups = []
    for key in partition.keys():
        if partition[key] not in groups:
            groups.append(partition[key])
    partition_groups = {group_i: [] for group_i in groups}
    for key in partition.keys():
        partition_groups[partition[key]].append(key)
    group_len_max = len(graph_nx) // client_num
    for group_i in groups:
        while len(partition_groups[group_i]) > group_len_max:
            long_group = list.copy(partition_groups[group_i])
            partition_groups[group_i] = list.copy(long_group[:group_len_max])
            new_grp_i = max(groups) + 1
            groups.append(new_grp_i)
            partition_groups[new_grp_i] = long_group[group_len_max:]
    len_list = []
    for group_i in groups:
        len_list.append(len(partition_groups[group_i]))
    len_dict = {}
    for i in range(len(groups)):
        len_dict[groups[i]] = len_list[i]
    sort_len_dict = {k: v for k, v in sorted(len_dict.items(), key=lambda item: item[1], reverse=True)}
    owner_node_ids = {owner_id: [] for owner_id in range(client_num)}
    owner_nodes_len = len(graph_nx) // client_num
    owner_list = [i for i in range(client_num)]
    owner_ind = 0
    flag = 0
    for group_i in sort_len_dict.keys():
        while len(owner_node_ids[owner_list[owner_ind]]) > owner_nodes_len:
            owner_list.remove(owner_list[owner_ind])
            owner_ind = owner_ind % len(owner_list)
        k = 0
        while len(owner_node_ids[owner_list[owner_ind]]) + len(partition_groups[group_i]) > owner_nodes_len + 1:
            k += 1
            owner_ind = (owner_ind + 1) % len(owner_list)
            if k == len(owner_list):
                owner_node_ids[owner_list[owner_ind]] += partition_groups[group_i]
                flag = 1
                break
        if(flag == 0):
            owner_node_ids[owner_list[owner_ind]] += partition_groups[group_i]
        else:
            flag = 0
    node_dict = owner_node_ids

    subgraph_list = construct_subgraph_dict_from_node_dict(
        G=G,
        num_clients=client_num,
        node_dict=node_dict,
        graph_nx=graph_nx,
        train=train_rate,
        val=val_rate,
        test=test_rate
    )

    return subgraph_list
