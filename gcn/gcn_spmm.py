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

    # def forward_(self, input, adj):
    #     support = flow.mm(input, self.weight)
    #     output = flow.spmm(adj, support)
    #     if self.bias is not None:
    #         return output + self.bias
    #     else:
    #         return output

    def forward(self, input, adj_coo):
        support = flow.mm(input, self.weight)
        output = flow._C.spmm_coo(adj_coo[0], adj_coo[1], adj_coo[2], adj_coo[3], adj_coo[4], support)
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

    def forward(self, x, adj_coo):
        x = F.relu(self.gc1(x, adj_coo))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj_coo)
        return F.log_softmax(x, dim=1)


