from __future__ import division
from __future__ import print_function
from errno import EMEDIUMTYPE

import time
import argparse
import numpy as np
import math
import oneflow as flow

from utils import load_data, accuracy
from gcn_spmm import GCN, GraphConvolution

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
device = "cuda:0"
np.random.seed(args.seed)
flow.manual_seed(args.seed)
flow.cuda.manual_seed(args.seed)

adj, features, labels, idx_train, idx_val, idx_test = load_data()
csrRowInd, csrColInd, csrValues = adj
num_nodes, feat_dim = features.shape
hid_dim = args.hidden
cls_dim = labels.max().item() + 1
rows = num_nodes
cols = num_nodes
batch_size = num_nodes
num_edges = len(csrValues)
train_num = len(idx_train)
num_layers = 2
features = features.to(device)
labels = labels.to(device)
csrRowInd = csrRowInd.to(device)
csrColInd = csrColInd.to(device)
csrValues = csrValues.to(device)
adj_csr = [csrRowInd, csrColInd, csrValues, rows, cols] 


embedding = flow.nn.Embedding(num_nodes, feat_dim, max_norm=True).to(device)
gcn = GCN(nfeat=feat_dim,  #1433
            nhid=args.hidden,  # 16
            nclass=labels.max().item() + 1,  # 7
            dropout=args.dropout) 
gcn = gcn.to(device)
loss_fn = flow.nn.NLLLoss() 

class GCNModule(flow.nn.Module):
    def __init__(self):
        super(GCNModule, self).__init__()
        self.embedding = embedding
        self.gcn = gcn
    
    def forward(self, ids):
        emb = self.embedding(ids)
        out = self.gcn(emb, adj_csr)
        return out


ids = np.arange(0, num_nodes, 1, dtype=np.int32)
ids_tensor = flow.tensor(ids, requires_grad=False).to(device)
print("embedding", embedding(ids_tensor).shape)
embedding.weight.requires_grad = False
module = GCNModule()
optimizer = flow.optim.Adam(module.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
for epoch in range(args.epochs):
    module.train()
    optimizer.zero_grad()
    output = module(ids_tensor)
    loss_train = loss_fn(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()