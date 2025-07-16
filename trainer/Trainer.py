import numpy as np
import copy

class Trainer:
    def __init__(self, server, clients, args):
        self.args = args
        self.server = server
        self.num_clients = len(clients)
        self.clients = clients
        self.rounds = args.rounds

    def train(self):
        m = max(int(self.args.client_sample_ratio * self.args.clients_nums), 1)
        for round in range(self.rounds):
            print("round: ", round)
            # 获取全局模型参数,发送给每个客户端
            global_state_dict = copy.deepcopy(self.server.state_dict())
            for i in range(self.args.clients_nums):
                self.clients[i].update(global_state_dict)

            selected_clients = np.random.choice(range(self.args.clients_nums), m, replace=False)
            selected_clients = np.sort(selected_clients)
            for i in selected_clients:
                self.clients[i].train_local_model()

            self.server.avg(self.clients, selected_clients)
            
            self.server.test_global(self.clients)

