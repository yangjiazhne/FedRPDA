from data.splitter import LouvainSplitter

def split_communities(G, clients):
    subgraph_list = LouvainSplitter(
            G=G,
            client_num=clients,
            train_rate=0.4,
            val_rate=0.2,
            test_rate=0.4,
        )

    return subgraph_list