from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import torch

import oneflow as flow

from utils import load_data, accuracy
from gcn_spmm import GCN

# Training settings
parser = argparse.ArgumentParser()

parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()

np.random.seed(args.seed)
flow.manual_seed(args.seed)
flow.cuda.manual_seed(args.seed)

# Load from tensors, fixed
print("load tensors")
ts = torch.load("/home/chenyuang/pygcn-kipf/pygcn/tensor.pt")
device = flow.device("cuda:0")

adj = ts["adj"].cpu()
features = flow.utils.tensor.from_torch(ts["features"].cpu()).to(device)
labels  = flow.utils.tensor.from_torch(ts["labels"].cpu()).to(device)
idx_train = flow.utils.tensor.from_torch(ts["idx_train"].cpu()).to(device)
idx_val = flow.utils.tensor.from_torch(ts["idx_val"].cpu()).to(device)
idx_test = flow.utils.tensor.from_torch(ts["idx_test"].cpu()).to(device)

## load from data, shuffle
#adj, features, labels, idx_train, idx_val, idx_test = load_data()
#csrRowInd, csrColInd, csrValues = adj

num_nodes, feat_dim = features.shape
rows = num_nodes
cols = num_nodes
batch_size = num_nodes

csrRowInd = flow.utils.tensor.from_torch(adj.crow_indices())
csrColInd = flow.utils.tensor.from_torch(adj.col_indices())
csrValues = flow.utils.tensor.from_torch(adj.values())
num_edges = len(csrValues)
train_num = len(idx_train)
csrRowInd = csrRowInd.to(device, dtype=flow.int32)
csrColInd = csrColInd.to(device, dtype=flow.int32)
csrValues = csrValues.to(device)

adj_csr = [csrRowInd, csrColInd, csrValues, rows, cols] 
num_nodes, feat_dim = features.shape
rows = num_nodes
cols = num_nodes
batch_size = num_nodes
num_edges = len(csrValues)
train_num = len(idx_train)


print("num_nodes, feat_dim, rows, cols, num_edges, train_num, num_class")
print(num_nodes, feat_dim, rows, cols, num_edges, train_num, labels.max().item() + 1)

device = flow.device("cuda:0")
print("Using {} device".format(device))

# Model and optimizer
model = GCN(nfeat=feat_dim,  #2708
            nhid=args.hidden,  # 16
            nclass=labels.max().item() + 1,  # 7
            dropout=args.dropout)   
loss_fn = flow.nn.NLLLoss().to(device)
optimizer = flow.optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
# if args.cuda:
model.to(device)


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj_csr)
    loss_train = loss_fn(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    loss_val = loss_fn(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    if(epoch % 10 ==0):
        print('Epoch: {:04d}'.format(epoch+1),
            'loss_train: {:.4f}'.format(loss_train.item()),
            'acc_train: {:.4f}'.format(acc_train.item()),
            'loss_val: {:.4f}'.format(loss_val.item()),
            'acc_val: {:.4f}'.format(acc_val.item()),
            'time: {:.4f}s'.format(time.time() - t)) 

def test():
    model.eval()
    output = model(features, adj_csr)
    loss_test = loss_fn(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
""" 
state_dict = model.state_dict()
flow.save(state_dict, './state') 

#model.load_state_dict(flow.load("/home/chenyuang/pygcn-kipf/pygcn/"))
#model.to(device)
"""


