import torch
import torch.nn.functional as F
import os
import sys
import errno
import os.path as osp
from sklearn.metrics import (average_precision_score,
                             f1_score,
                             precision_score,
                             recall_score, 
                             roc_auc_score,
                             confusion_matrix)

def index_to_mask(index_list, length):
    mask = torch.zeros(length, dtype=torch.bool)
    mask[index_list] = True
    return mask

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def train_batch(model, train_loader, optimizer, criterion, device, neg_prototype, pos_prototype, lamda, m=0.1):
    model.train()
    total_loss1 = total_loss2 = total_correct = total_examples = 0
    
    running_mu = None
    running_sigma = None    
    
    for batch in train_loader:
        optimizer.zero_grad()

        y = batch.y[:batch.batch_size].to(device)
        outputs = model(batch.x, batch.edge_index.to(device))
        out = outputs['logits'][:batch.batch_size]
        local_features = outputs['feature'][:batch.batch_size].detach()
        
        loss1 = criterion(out, y)
        
        train_pos = y == 1
        train_neg = y == 0
        pos_index = train_pos.nonzero().flatten()
        neg_index = train_neg.nonzero().flatten()
        
        n_pos = len(pos_index)
        n_neg = len(neg_index)
        n_perturb = max(n_neg - n_pos, 0) 

        if n_pos == 0 or neg_prototype==None or pos_prototype==None:
            loss2 = torch.tensor(0., device=device)
        else:
            pos_feat = local_features[pos_index]
            batch_mu = pos_feat.mean(dim=0)
            batch_sigma = pos_feat.std(dim=0, unbiased=False)
            if running_mu is None:
                running_mu = batch_mu
                running_sigma = batch_sigma
            else:
                running_mu = (1 - m) * running_mu + m * batch_mu
                running_sigma = (1 - m) * running_sigma + m * batch_sigma
            if n_perturb > 0:
                eps = torch.randn(n_perturb, batch_mu.shape[0], device=device)
                perturbed_feat = running_mu + running_sigma * eps
                y_perturb = torch.ones(n_perturb, dtype=torch.long, device=device)
            else:
                perturbed_feat = torch.empty((0, batch_mu.shape[0]), device=device)
                y_perturb = torch.empty((0,), dtype=torch.long, device=device)
            neg_feat = local_features[neg_index]
            y_neg = torch.zeros(n_neg, dtype=torch.long, device=device)
            y_pos = torch.ones(n_pos, dtype=torch.long, device=device)
            all_feat = torch.cat([neg_feat, pos_feat, perturbed_feat], dim=0)
            all_y = torch.cat([y_neg, y_pos, y_perturb], dim=0)

            pos_proto = pos_prototype.expand(all_feat.shape)
            neg_proto = neg_prototype.expand(all_feat.shape)
            diff_pos = -F.pairwise_distance(all_feat, pos_proto)
            diff_neg = -F.pairwise_distance(all_feat, neg_proto)
            diff = torch.cat([diff_neg.view(-1,1), diff_pos.view(-1,1)], dim=1)
            loss2 = F.cross_entropy(diff, all_y)

        loss = loss1 + lamda * loss2
        
        loss.backward()
        optimizer.step()

        total_loss1 += float(loss1) * batch.batch_size
        total_loss2 += float(loss2) * batch.batch_size
        total_correct += int((out.argmax(dim=-1) == y).sum())
        total_examples += batch.batch_size

    return total_loss1 / total_examples, total_loss2 / total_examples, total_correct / total_examples

@torch.no_grad()
def get_local_prop(model, train_loader, device):
    model.eval()
    
    pos_features = []
    neg_features = []

    for batch in train_loader:
        y = batch.y[:batch.batch_size].to(device)
        outputs = model(batch.x, batch.edge_index.to(device))
        local_features = outputs['feature'][:batch.batch_size].detach()

        train_pos = local_features[y == 1]
        train_neg = local_features[y == 0]

        pos_features.append(train_pos)
        neg_features.append(train_neg)

    all_pos_features = torch.cat(pos_features, dim=0)
    all_neg_features = torch.cat(neg_features, dim=0)

    pos = torch.mean(all_pos_features, dim=0, keepdim=True)
    neg = torch.mean(all_neg_features, dim=0, keepdim=True)

    return pos, neg

@torch.no_grad()
def test_batch(model, graph, subgraph_loader, device):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    out = model.inference(graph.x, subgraph_loader, device)
    pred = out.argmax(dim=-1)

    test_loss = criterion(out[graph.test_mask], graph.y[graph.test_mask])

    test_correct = pred[graph.test_mask] == graph.y[graph.test_mask]
    test_acc = int(test_correct.sum()) / int(graph.test_mask.sum())

    val_correct = pred[graph.val_mask] == graph.y[graph.val_mask]
    val_acc = int(val_correct.sum()) / int(graph.val_mask.sum())

    conf_matrix = confusion_matrix(graph.y[graph.test_mask].cpu().numpy(), pred[graph.test_mask].cpu().numpy())
    f1 = f1_score(y_true=graph.y[graph.test_mask].cpu().numpy(), y_pred=pred[graph.test_mask].cpu().numpy(),
                  average='macro', zero_division=1)
    precision = precision_score(y_true=graph.y[graph.test_mask].cpu().numpy(), y_pred=pred[graph.test_mask].cpu().numpy(),
                                average='macro',
                                zero_division=1)
    recall = recall_score(y_true=graph.y[graph.test_mask].cpu().numpy(), y_pred=pred[graph.test_mask].cpu().numpy(),
                          average='macro', zero_division=1)
    y_scores = out[graph.test_mask][:, 1]
    auc = roc_auc_score(graph.y[graph.test_mask].detach().cpu().numpy(), y_scores.detach().cpu().numpy())
    aps = average_precision_score(graph.y[graph.test_mask].detach().cpu().numpy(), y_scores.detach().cpu().numpy())
    return test_loss, val_acc, test_acc, f1, precision, recall, auc, aps, conf_matrix

def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

class Logger(object):
    """
    Write console output to external text file.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """

    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()