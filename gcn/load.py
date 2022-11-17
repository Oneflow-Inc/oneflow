
import time
import argparse
import numpy as np
import oneflow as flow
from scipy.sparse import csr_matrix
import math
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import load_data, accuracy
from gcn_spmm import GCN

torch.backends.cuda.matmul.allow_tf32 = True


parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
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
args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)
flow.manual_seed(args.seed)
if args.cuda:
    flow.cuda.manual_seed(args.seed)


print("load tensors")
ts = torch.load("/home/chenyuang/pygcn-kipf/pygcn/tensor.pt")
device = flow.device("cuda:0")

adj = ts["adj"].cpu()
features = flow.utils.tensor.from_torch(ts["features"].cpu()).to(device)
labels  = flow.utils.tensor.from_torch(ts["labels"].cpu()).to(device)
idx_train = flow.utils.tensor.from_torch(ts["idx_train"].cpu()).to(device)
idx_val = flow.utils.tensor.from_torch(ts["idx_val"].cpu()).to(device)
idx_test = flow.utils.tensor.from_torch(ts["idx_test"].cpu()).to(device)

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
model_infer = GCN(nfeat=feat_dim,  #2708
            nhid=args.hidden,  # 16
            nclass=labels.max().item() + 1,  # 7
            dropout=args.dropout)  
loss_fn = flow.nn.NLLLoss().to(device)

print("num_nodes, feat_dim, rows, cols, num_edges, train_num, num_class")
print(num_nodes, feat_dim, rows, cols, num_edges, train_num, labels.max().item() + 1)

def infer(model_test):
    model_test.eval()
    output = model_test(features, adj_csr)
    loss_test = loss_fn(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Infer set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

parameters = torch.load("/home/chenyuang/pygcn-kipf/pygcn/pygcn.pt")
new_parameters = dict()
for key,value in parameters['model_state_dict'].items():
     if "num_batches_tracked" not in key:
          val = value.detach().cpu().numpy()
          new_parameters[key] = val 


#print('oneflow output\n', output[0])
def train(mymodel):
    t = time.time()
    mymodel.train()
    optimizer = flow.optim.Adam(mymodel.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

    optimizer.zero_grad()
    output = mymodel(features, adj_csr)
    print("------------")
    print(mymodel.gc1.weight)
    loss_train = loss_fn(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    print("------------")
    print(mymodel.gc1.weight)
    loss_val = loss_fn(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])

model_infer.load_state_dict(new_parameters)
model_infer.to('cuda:0')
for epoch in range(args.epochs):
    print("@@@@@@@@@@@@@@@@@ epoch ", epoch)
    train(model_infer)


""" support_np = support.detach().cpu().numpy()
torch_y = torch.spmm(adj, torch.FloatTensor((support_np)))
#print('torch output\n', output2[0])


numpy_y = (
     csr_matrix((csrValues.detach().cpu().numpy(),
                csrColInd.detach().cpu().numpy(),
                csrRowInd.detach().cpu().numpy()),
                shape=(rows, cols))
                 * support ) """
     
#print('scipy output\n', output3[0])

""" np.save('csrRowInd.npy', csrRowInd.detach().cpu().numpy()) 
np.save('csrColInd.npy', csrColInd.detach().cpu().numpy()) 
np.save('csrValues.npy', csrValues.detach().cpu().numpy()) 
np.save('support.npy', support.detach().cpu().numpy()) 
np.save('dimension.npy', np.asarray([rows, cols]))  """

