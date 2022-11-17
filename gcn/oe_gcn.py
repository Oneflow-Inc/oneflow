from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import oneflow as flow
from oneflow.one_embedding import make_persistent_table_reader
from oneflow.one_embedding import make_persistent_table_writer

from utils import load_data, accuracy
from gcn_spmm import GCN, GraphConvolution

device = "cuda:0"

class OneEmbedding(flow.nn.Module):
    def __init__(
        self,
        embedding_size,
        table_size,
        name="gcn_embedding",
        persistent_path="./gcn_embedding",
        store_type="cached_ssd",
        cache_memory_budget_mb=256,
        key_type=flow.int32,
        data_type=flow.float
    ):

        scale = np.sqrt(1 / np.array(embedding_size))
        tables = [
            flow.one_embedding.make_table_options(
                flow.one_embedding.make_uniform_initializer(low=-scale, high=scale)
            )
        ]
        if store_type == "device_mem":
            store_options = flow.one_embedding.make_device_mem_store_options(
                persistent_path=persistent_path, capacity=table_size
            )
        elif store_type == "cached_host_mem":
            assert cache_memory_budget_mb > 0
            store_options = flow.one_embedding.make_cached_host_mem_store_options(
                cache_budget_mb=cache_memory_budget_mb,
                persistent_path=persistent_path,
                capacity=table_size,
            )
        elif store_type == "cached_ssd":
            assert cache_memory_budget_mb > 0
            store_options = flow.one_embedding.make_cached_ssd_store_options(
                cache_budget_mb=cache_memory_budget_mb,
                persistent_path=persistent_path,
                capacity=table_size,
                size_factor=1,
                physical_block_size=512
            )
        else:
            raise NotImplementedError("not support", store_type)

        super(OneEmbedding, self).__init__()
        self.one_embedding = flow.one_embedding.MultiTableEmbedding(
            name=name,
            embedding_dim=embedding_size,
            dtype=data_type,
            key_type=key_type,
            tables=tables,
            store_options=store_options,
        )

    def forward(self, ids):
        return self.one_embedding.forward(ids)

class GCNModule(flow.nn.Module):
    def __init__(
        self,
        feat_dim,
        num_nodes,
        hid_dim,
        cls_dim
    ):
        super().__init__()

        self.embedding_layer = OneEmbedding(
            embedding_size=feat_dim, #1433
            table_size=num_nodes     
        )
        #self.embedding_layer.requires_grad = False
        self.gcn = GCN(nfeat=feat_dim,  #1433
                    nhid=hid_dim,  # 16
                    nclass=cls_dim,  # 7
                    dropout=0.5)

    def forward(self, inputs, adj):
        with flow.no_grad():
            emb = self.embedding_layer(inputs)
        out = self.gcn(emb, adj)
        return out

class GCNGraph(flow.nn.Graph):
    def __init__(        
        self,
        adj,
        loss_fn,
        gcn_module):
        super().__init__()
        self.gcn_module = gcn_module
        self.add_optimizer(
            flow.optim.Adam(self.gcn_module.gcn.parameters(),
                       lr=0.01, weight_decay=5e-4)
        )
        self.add_optimizer(
            flow.optim.SGD(self.gcn_module.embedding_layer.parameters(),
                       lr=0.1, momentum=0.0)
        )
        self.loss_fn = loss_fn
        self.adj = adj

 
    def build(self, ids, labels, idx_train):
        output = self.gcn_module(ids, self.adj)
        loss = self.loss_fn(output[idx_train], labels[idx_train])
        acc = accuracy(output[idx_train], labels[idx_train])
        loss.backward()
        return loss, acc

class ForwardGCNGraph(flow.nn.Graph):
    def __init__(self, adj, gcn_module, loss_fn):
        super(ForwardGCNGraph, self).__init__()
        self.adj = adj
        self.gcn_module = gcn_module
        self.loss_fn = loss_fn

    def build(self, ids, labels, idx_test):
        output = self.gcn_module(ids, self.adj)
        loss = self.loss_fn(output[idx_test], labels[idx_test])
        acc = accuracy(output[idx_test], labels[idx_test])
        return loss, acc
        
def trainGCNModule(feat_dim, num_nodes, hid_dim, cls_dim, datatuple, epochs, save):
        module = GCNModule(    
            feat_dim=feat_dim,
            num_nodes=num_nodes,
            hid_dim=hid_dim,
            cls_dim=cls_dim)
        loss_fn = flow.nn.NLLLoss().to(device)
        optimizer = flow.optim.Adam(module.parameters(),
                         lr=0.01, weight_decay=5e-4)
        optimizer = flow.one_embedding.Optimizer(
                optimizer, embeddings=[module.embedding_layer.one_embedding])
        module.to(device)

        print("eager running")
        ids_tensor, adj, labels, idx_train = datatuple
        for epoch in range(epochs):
            module.train()
            optimizer.zero_grad()
            output = module(ids_tensor, adj)
            loss_train = loss_fn(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
    
        if(save):
            state_dict = module.state_dict()
            for key in state_dict.keys():
                print(key)
            flow.save(state_dict, './gcn_module_state') 

def runGCNGraph(feat_dim, loss_fn, num_nodes, hid_dim, cls_dim, datatuple, args):
        ids_tensor, adj, labels, idx_train, idx_val, idx_test = datatuple
      
        module = GCNModule(    
            feat_dim=feat_dim,
            num_nodes=num_nodes,
            hid_dim=hid_dim,
            cls_dim=cls_dim)
        
        module.to(device)

        gcn_graph = GCNGraph(
            adj=adj,
            gcn_module=module,
            loss_fn=loss_fn)

        fwd_gcn_graph = ForwardGCNGraph(
            adj=adj,
            gcn_module=module,
            loss_fn=loss_fn)


        gcn_graph.load_state_dict(state_dict = flow.load("./gcn_graph_state"))  

        for name, param in module.named_parameters():
            if(param.requires_grad):
                print(name)       
        for epoch in range(args.epochs):
            loss, acc = gcn_graph(ids_tensor, labels, idx_train)
            if(epoch % 10 == 0):
                print("train accuracy: ", acc, ", epoch ", epoch)
        loss, acc = fwd_gcn_graph(ids_tensor, labels, idx_test)
        print("test accuracy: ", acc)
        # loss_test = loss_fn(output[idx_test], labels[idx_test])
        # acc_test = accuracy(output[idx_test], labels[idx_test])
        # print("Test set results:",
        #     "loss= {:.4f}".format(loss_test.item()),
        #     "accuracy= {:.4f}".format(acc_test.item()))

        if(args.save):
            tables = ["./gcn_embedding/0-1"]
            state_dict = gcn_graph.state_dict()
            snapshot = state_dict["gcn_module"]['embedding_layer.one_embedding.OneEmbeddingSnapshot']
            with make_persistent_table_reader(tables, snapshot, flow.int32, flow.float, 1433, 512) as reader:
                for ks, vs in reader:
                    print(ks.shape, vs.shape)
                    print(vs[:10])
                    break
            #flow.save(state_dict, './gcn_graph_state') 

def load_module_state(eager, graph, read, write):
        if(eager):
            state_dict = flow.load("./gcn_module_state")
        if(graph):
            state_dict = flow.load("./gcn_graph_state")
        snapshot = state_dict["gcn_module"]['embedding_layer.one_embedding.OneEmbeddingSnapshot']
        print(snapshot)
        print("--------")
        tables = ["./gcn_embedding/0-1"]
        if(read):
            with make_persistent_table_reader(tables, snapshot, flow.int32, flow.float, 1433, 512) as reader:
                for ks, vs in reader:
                    print(ks.shape, vs.shape)
                    print(vs[:10])
                    break
            
        if(write):
            adj, features, labels, idx_train, idx_val, idx_test = load_data()
            num_nodes, feat_dim = features.shape
            with make_persistent_table_writer(tables, snapshot, flow.int32, flow.float, 1433, 512) as writer:
                keys = np.arange(num_nodes).astype(np.int32)
                values = features.detach().cpu().numpy()
                writer.write(keys, values)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train.')
    parser.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--graph', action='store_true', default=False,
                        help='run in Graph mode')
    parser.add_argument('--eager', action='store_true', default=False,
                        help='run in Eager mode')
    parser.add_argument('--save', action='store_true', default=False,
                        help='save the model')
    parser.add_argument('--load', action='store_true', default=False,
                        help='load the model')
    parser.add_argument('--read', action='store_true', default=False,
                        help='read one_embedding')
    parser.add_argument('--write', action='store_true', default=False,
                        help='write one_embedding') 
    args = parser.parse_args()
    np.random.seed(args.seed)
    flow.manual_seed(args.seed)
    flow.cuda.manual_seed(args.seed)

     # tables = ['./gcn_embedding/0-1']
    # with make_persistent_table_reader(tables, "2022-11-01-03-55-34-201609", flow.int32, flow.float, 1536) as reader:
    #     for ks, vs in reader:
    #         print(ks.shape, vs.shape)
    #         print(ks[:10])
    #         print(vs[0])
    #         break
   
    ## load model
    if(args.load): 
        load_module_state(args.eager, args.graph, args.read, args.write)
    ## train model
    else:
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
        idx_train = idx_train.to(device)
        csrRowInd = csrRowInd.to(device)
        csrColInd = csrColInd.to(device)
        csrValues = csrValues.to(device)
        adj_csr = [csrRowInd, csrColInd, csrValues, rows, cols] # [tensor, tensor, tensor, int32, int32]
        t_total = time.time()
        count = 0
        ids = np.arange(0, num_nodes, 1, dtype=np.int32)
        ids_tensor = flow.tensor(ids, requires_grad=False).to(device)
        datatuple = (ids_tensor, adj_csr, labels, idx_train, idx_val, idx_test)

        if(args.eager):
            trainGCNModule(
                num_nodes=num_nodes,
                feat_dim=feat_dim,
                hid_dim=hid_dim,
                cls_dim=cls_dim,
                datatuple=datatuple,
                args = args
            )
        if(args.graph):
            runGCNGraph(
                num_nodes=num_nodes,
                loss_fn=flow.nn.NLLLoss(),
                feat_dim=feat_dim,
                hid_dim=hid_dim,
                cls_dim=cls_dim,
                datatuple=datatuple,
                args = args
            )

