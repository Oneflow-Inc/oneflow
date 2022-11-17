import math
import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.functional as F
import numpy as np

# kipf 
class GraphConvolution(nn.Module):
    def __init__(self, infeat, outfeat, bias = True):
        super(GraphConvolution, self).__init__()
        self.infeat = infeat
        self.outfeat = outfeat
        self.weight = nn.Parameter(flow.FloatTensor(infeat, outfeat))
        if bias:
            self.bias = nn.Parameter(flow.FloatTensor(outfeat))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj_csr):
        support = flow.mm(input, self.weight)
        output = flow._C.spmm_csr(adj_csr[0], adj_csr[1], adj_csr[2], adj_csr[3], adj_csr[4], support)
       # output = flow.mm(input, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GCN(nn.Module): 
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj_csr):
        x = F.relu(self.gc1(x, adj_csr))
        x = F.dropout(x, self.dropout, training=self.training) 
        x = self.gc2(x, adj_csr)

        return F.log_softmax(x, dim=1)


