from __future__ import division
from __future__ import print_function

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

#
embedding_size = feat_dim
scale = np.sqrt(1 / np.array(embedding_size))
tables = [
    flow.one_embedding.make_table_options(
        flow.one_embedding.make_uniform_initializer(low=-scale, high=scale))
]


store_options = flow.one_embedding.make_cached_ssd_store_options(
cache_budget_mb=256,
persistent_path="./eager_gcn_embedding",
capacity=num_nodes,
size_factor=3,
physical_block_size=512
)


class OneEmbedding(flow.nn.Module):
    def __init__(self) -> None:
        super(OneEmbedding, self).__init__()
        self.one_embedding = flow.one_embedding.MultiTableEmbedding(
                            name="eager_gcn_embedding",
                            embedding_dim=embedding_size,
                            dtype=flow.float,
                            key_type=flow.int32,
                            tables=tables,
                            store_options=store_options
                        )
    def forward(self, ids):
        return self.one_embedding.forward(ids)

 
class GCNModule(flow.nn.Module):
    def __init__(self):
        super(GCNModule, self).__init__()
        self.embedding_layer = OneEmbedding()
        self.gcn = GCN(nfeat=feat_dim, nhid=args.hidden, nclass=labels.max().item() + 1, dropout=args.dropout) 


    def forward(self, ids):
        emb = self.embedding_layer(ids)
        output = self.gcn(emb, adj_csr)
        return output

# input and optimizer
ids = np.arange(0, num_nodes, 1, dtype=np.int32)
ids_tensor = flow.tensor(ids, requires_grad=False).to(device)
loss_fn = flow.nn.NLLLoss() 

model = GCNModule()
optimizer = flow.optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
optimizer = flow.one_embedding.Optimizer(optimizer, [model.embedding_layer.one_embedding])
model.train()
model.to(device)


for epoch in range(args.epochs):
    model.train()
    model.embedding_layer.requires_grad = False
    optimizer.zero_grad()
    output = model(ids_tensor)
    loss_train = loss_fn(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()