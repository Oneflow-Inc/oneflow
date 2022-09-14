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
# adj, features, labels, idx_train, idx_val, idx_test = load_data()
# cooRowInd, cooColInd, cooValues = adj


NODE_NUM, FEATURE_DIM = features.shape
rows = NODE_NUM
cols = NODE_NUM
BATCH_SIZE = NODE_NUM
HIDDEN_SIZE = 16
CLASS = 7
EDGE_NUM = len(cooRowInd)
TRAIN_NUM = len(idx_train)

#len(cooRowInd) == len(cooColInd) == len(cooValues) == EDGE_NUM
print(cooRowInd.dtype)
print(cooColInd.dtype)
print(cooValues.dtype)
print(features.dtype)

print(NODE_NUM, FEATURE_DIM, rows, cols, EDGE_NUM, TRAIN_NUM)


input = flow.nn.Parameter(flow.FloatTensor(4, 4))

weight = flow.nn.Parameter(flow.FloatTensor(4, 3))
support = flow.mm(input, weight)
print(support.shape[0])


# Model and optimizer
model = GCN(nfeat=FEATURE_DIM,
            nhid=HIDDEN_SIZE,
            nclass=CLASS,
            dropout=0.5)
optimizer = optim.Adam(model.parameters(),
                       lr=0.01, weight_decay=5e-4)
# if args.cuda:
# model.cuda()
# features = features.cuda()
# adj = adj.cuda()
# labels = labels.cuda()
# idx_train = idx_train.cuda()
# idx_val = idx_val.cuda()
# idx_test = idx_test.cuda()


# def train(epoch):
#     t = time.time()
#     model.train()
#     optimizer.zero_grad()
#     output = model(features, adj)
#     loss_train = F.nll_loss(output[idx_train], labels[idx_train])
#     acc_train = accuracy(output[idx_train], labels[idx_train])
#     loss_train.backward()
#     optimizer.step()

#     # if not args.fastmode:
#     #     # Evaluate validation set performance separately,
#     #     # deactivates dropout during validation run.
#     #     model.eval()
#     #     output = model(features, adj)

#     loss_val = F.nll_loss(output[idx_val], labels[idx_val])
#     acc_val = accuracy(output[idx_val], labels[idx_val])
#     print('Epoch: {:04d}'.format(epoch+1),
#           'loss_train: {:.4f}'.format(loss_train.item()),
#           'acc_train: {:.4f}'.format(acc_train.item()),
#           'loss_val: {:.4f}'.format(loss_val.item()),
#           'acc_val: {:.4f}'.format(acc_val.item()),
#           'time: {:.4f}s'.format(time.time() - t))


# def test():
#     model.eval()
#     output = model(features, adj)
#     loss_test = F.nll_loss(output[idx_test], labels[idx_test])
#     acc_test = accuracy(output[idx_test], labels[idx_test])
#     print("Test set results:",
#           "loss= {:.4f}".format(loss_test.item()),
#           "accuracy= {:.4f}".format(acc_test.item()))


# Train model
# t_total = time.time()
# for epoch in range(args.epochs):
#     train(epoch)
# print("Optimization Finished!")
# print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# # Testing
# test()