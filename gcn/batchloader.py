import numpy as np
import dgl
import oneflow as flow
from ogb.nodeproppred import DglNodePropPredDataset

t2f = flow.utils.tensor.from_torch


def dataloader(mode="train", device="cuda"):

    # load and preprocess dataset
    print('Loading data')
    dataset = DglNodePropPredDataset('ogbn-arxiv')

    graph, node_labels = dataset[0]
    # Add reverse edges since ogbn-arxiv is unidirectional.
    graph = dgl.add_reverse_edges(graph)
    graph.ndata['label'] = node_labels[:, 0]
    node_features = graph.ndata['feat']
    num_features = node_features.shape[1]
    num_classes = (node_labels.max() + 1).item()
   
    idx_split = dataset.get_idx_split()
    train_nids = idx_split['train']
    valid_nids = idx_split['valid']
    test_nids = idx_split['test']
  
    sampler = dgl.dataloading.NeighborSampler([4, 4])
   
    if mode=="train" :
        loader = dgl.dataloading.DataLoader(
            # The following arguments are specific to DGL's DataLoader.
            graph,              # The graph
            train_nids,         # The node IDs to iterate over in minibatches
            sampler,            # The neighbor sampler
            device="cpu",      # Put the sampled MFGs on CPU or GPU
            # The following arguments are inherited from PyTorch DataLoader.
            batch_size=1024,    # Batch size
            shuffle=True,       # Whether to shuffle the nodes for every epoch
            drop_last=False,    # Whether to drop the last incomplete batch
            num_workers=0       # Number of sampler processes
        )
    elif mode == "valid":
        loader = dgl.dataloading.DataLoader(
            # The following arguments are specific to DGL's DataLoader.
            graph,              # The graph
            valid_nids,         # The node IDs to iterate over in minibatches
            sampler,            # The neighbor sampler
            device="cpu",      # Put the sampled MFGs on CPU or GPU
            # The following arguments are inherited from PyTorch DataLoader.
            batch_size=1024,    # Batch size
            shuffle=False,       # Whether to shuffle the nodes for every epoch
            drop_last=False,    # Whether to drop the last incomplete batch
            num_workers=0       # Number of sampler processes
        )
    else:
        print("undefined mode")
        return
            
    batchs = []
    for input_nodes, output_nodes, mfgs in iter(loader):
        input_nodes = t2f(input_nodes)
        output_nodes = t2f(output_nodes)
        blocks = []
        adjs   = []
        for mfg in mfgs:
            mfg_src_id = t2f(mfg.srcdata[dgl.NID]).to(device)
            mfg_src_feat = t2f(mfg.srcdata["feat"]).to(device)
            mfg_src_label = t2f(mfg.srcdata["label"]).to(device)
          
            mfg_dst_id = t2f(mfg.dstdata[dgl.NID]).to(device)
            mfg_dst_feat = t2f(mfg.dstdata["feat"]).to(device)
            mfg_dst_label = t2f(mfg.dstdata["label"]).to(device)
            mfg_src_id
            block = { 
                    "num_src_nodes": mfg.num_src_nodes(),
                    "num_dst_nodes": mfg.num_dst_nodes(),
                    "num_edges": mfg.num_edges(),
                    "src_id": mfg_src_id,
                    "src_feat":  mfg_src_feat,
                    "src_label": mfg_src_label,
                    "dst_id": mfg_dst_id,
                    "dst_feat": mfg_dst_feat,
                    "dst_label": mfg_dst_label,
            }
            mfg_edata = mfg.edata[dgl.EID]
            sg = dgl.block_to_graph(mfg)
            sg_adj = sg.adjacency_matrix(transpose = True).to_sparse_csr()
            adj = [ 
                    t2f(sg_adj.crow_indices()).to(device, dtype=flow.int32),
                    t2f(sg_adj.col_indices()).to(device, dtype=flow.int32),
                    t2f(sg_adj.values()).to(device, dtype=flow.float),
                    sg_adj.size()[0],
                    sg_adj.size()[1],
                ]
            blocks.append(block)
            adjs.append(adj)
        mini_batch = input_nodes, output_nodes, blocks, adjs
        batchs.append(mini_batch)
    print("batch size: ", len(batchs))

    return batchs, num_features, num_classes
