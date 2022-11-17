import dgl
import torch
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset

dataset = DglNodePropPredDataset('ogbn-arxiv')
device = 'cuda:7'      # change to 'cuda' for GPU

graph, node_labels = dataset[0]
# Add reverse edges since ogbn-arxiv is unidirectional.
graph = dgl.add_reverse_edges(graph)
graph.ndata['label'] = node_labels[:, 0]

node_features = graph.ndata['feat']
num_features = node_features.shape[1]
num_classes = (node_labels.max() + 1).item()
print('Number of classes:', num_classes)

idx_split = dataset.get_idx_split()
train_nids = idx_split['train']
valid_nids = idx_split['valid']
test_nids = idx_split['test']


sampler = dgl.dataloading.NeighborSampler([4, 4])
train_dataloader = dgl.dataloading.DataLoader(
    # The following arguments are specific to DGL's DataLoader.
    graph,              # The graph
    train_nids,         # The node IDs to iterate over in minibatches
    sampler,            # The neighbor sampler
    device=device,      # Put the sampled MFGs on CPU or GPU
    # The following arguments are inherited from PyTorch DataLoader.
    batch_size=1024,    # Batch size
    shuffle=True,       # Whether to shuffle the nodes for every epoch
    drop_last=False,    # Whether to drop the last incomplete batch
    num_workers=0       # Number of sampler processes
)



input_nodes, output_nodes, mfgs = example_minibatch = next(iter(train_dataloader))
print("To compute {} nodes' outputs, we need {} nodes' input features".format(len(output_nodes), len(input_nodes))) 

import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv, GraphConv

class Model(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(Model, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)
        self.h_feats = h_feats

    def forward(self, mfgs, x):
        # Lines that are changed are marked with an arrow: "<---"
        h_dst = x[:mfgs[0].num_dst_nodes()]  # <---
        h = self.conv1(mfgs[0], (x, h_dst))  # <---
        h = F.relu(h)
        h_dst = h[:mfgs[1].num_dst_nodes()]  # <---
        h = self.conv2(mfgs[1], (h, h_dst))  # <---
        return h

model = Model(num_features, 128, num_classes).to(device)

opt = torch.optim.Adam(model.parameters())

valid_dataloader = dgl.dataloading.DataLoader(
    graph, valid_nids, sampler,
    batch_size=1024,
    shuffle=False,
    drop_last=False,
    num_workers=0,
    device=device
)

import tqdm
import sklearn.metrics

best_accuracy = 0
best_model_path = 'model.pt'
for epoch in range(100):
    model.train()

    with tqdm.tqdm(train_dataloader) as tq:
        for step, (input_nodes, output_nodes, mfgs) in enumerate(tq):
            # feature copy from CPU to GPU takes place here
            inputs = mfgs[0].srcdata['feat']
            labels = mfgs[-1].dstdata['label']

            predictions = model(mfgs, inputs)

            loss = F.cross_entropy(predictions, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()

            accuracy = sklearn.metrics.accuracy_score(labels.cpu().numpy(), predictions.argmax(1).detach().cpu().numpy())

            tq.set_postfix({'loss': '%.03f' % loss.item(), 'acc': '%.03f' % accuracy}, refresh=False)

model.eval()

predictions = []
labels = []
with tqdm.tqdm(valid_dataloader) as tq, torch.no_grad():
    for input_nodes, output_nodes, mfgs in tq:
        inputs = mfgs[0].srcdata['feat']
        labels.append(mfgs[-1].dstdata['label'].cpu().numpy())
        predictions.append(model(mfgs, inputs).argmax(1).cpu().numpy())
    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)
    accuracy = sklearn.metrics.accuracy_score(labels, predictions)
    print('Epoch {} Validation Accuracy {}'.format(epoch, accuracy))
    # if best_accuracy < accuracy:
    #     best_accuracy = accuracy
    #     torch.save(model.state_dict(), best_model_path)

    # Note that this tutorial do not train the whole model to the end.
    


######################################################################
# Conclusion
# ----------
#
# In this tutorial, you have learned how to train a multi-layer GraphSAGE
# with neighbor sampling.
#
# What’s next?
# ------------
#
# -  :doc:`Stochastic training of GNN for link
#    prediction <L2_large_link_prediction>`.
# -  :doc:`Adapting your custom GNN module for stochastic
#    training <L4_message_passing>`.
# -  During inference you may wish to disable neighbor sampling. If so,
#    please refer to the :ref:`user guide on exact offline
#    inference <guide-minibatch-inference>`.
#


# Thumbnail credits: Stanford CS224W Notes
# sphinx_gallery_thumbnail_path = '_static/blitz_1_introduction.png'
