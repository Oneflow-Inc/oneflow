from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import oneflow as flow
import oneflow.nn.functional as F
import oneflow.optim as optim

from utils import load_data, accuracy
from gcn_spmm import GCN

# Training settings
# parser = argparse.ArgumentParser()
# parser.add_argument('--no-cuda', action='store_true', default=False,
#                     help='Disables CUDA training.')
# parser.add_argument('--fastmode', action='store_true', default=False,
#                     help='Validate during training pass.')
# parser.add_argument('--seed', type=int, default=42, help='Random seed.')
# parser.add_argument('--epochs', type=int, default=200,
#                     help='Number of epochs to train.')
# parser.add_argument('--lr', type=float, default=0.01,
#                     help='Initial learning rate.')
# parser.add_argument('--weight_decay', type=float, default=5e-4,
#                     help='Weight decay (L2 loss on parameters).')
# parser.add_argument('--hidden', type=int, default=16,
#                     help='Number of hidden units.')
# parser.add_argument('--dropout', type=float, default=0.5,
#                     help='Dropout rate (1 - keep probability).')

# args = parser.parse_args()
# args.cuda = not args.no_cuda and flow.cuda.is_available()

# np.random.seed(args.seed)
# flow.manual_seed(args.seed)
# if args.cuda:
#     flow.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()
cooRowInd, cooColInd, cooValues = adj


num_nodes, feat_dim = features.shape
rows = num_nodes
cols = num_nodes
batch_size = num_nodes
hidden_size = 16
class_size = 7
num_edges = len(cooRowInd)
train_num = len(idx_train)

#len(cooRowInd) == len(cooColInd) == len(cooValues) == num_edges


print("num_nodes, feat_dim, rows, cols, num_edges, train_num")
print(num_nodes, feat_dim, rows, cols, num_edges, train_num)

device = flow.device("cuda:1")
print("Using {} device".format(device))

# Model and optimizer
model = GCN(nfeat=feat_dim,
            nhid=hidden_size,
            nclass=class_size,
            dropout=0.5)
loss_fn = flow.nn.NLLLoss().to(device)
optimizer = optim.Adam(model.parameters(),
                       lr=0.01, weight_decay=5e-4)
# if args.cuda:


model.to(device)
features = features.to(device)
cooRowInd = cooRowInd.to(device)
cooColInd = cooColInd.to(device)
cooValues = cooValues.to(device)
labels = labels.to(device)
idx_train = idx_train.to(device)
idx_val = idx_val.to(device)
idx_test = idx_test.to(device)

adj_coo = [cooRowInd, cooColInd, cooValues, rows, cols]

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj_coo)
    loss_train = loss_fn(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    # if not args.fastmode:
    #     # Evaluate validation set performance separately,
    #     # deactivates dropout during validation run.
    #     model.eval()
    #     output = model(features, adj)

    loss_val = loss_fn(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj_coo)
    loss_test = loss_fn(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
for epoch in range(200):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# # Testing
test()