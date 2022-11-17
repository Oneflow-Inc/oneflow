import os
import dgl
import numpy as np
import torch
import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.functional as F
import oneflow.optim as optim
import oneflow.multiprocessing as mp
import time
import argparse
import tqdm
import sklearn.metrics
from batchloader import dataloader

from gcn_spmm import GCN, GraphConvolution

class BlockGCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(BlockGCN, self).__init__()
        self.gc1 = GraphConvolution(in_feats, h_feats)
        self.gc2 = GraphConvolution(h_feats, num_classes)

    def forward(self, x, adjs):
        x = F.relu(self.gc1(x, adjs[0]))
        x = self.gc2(x, adjs[1])
        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default='mixed', choices=['cpu', 'mixed', 'puregpu'],
                        help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
                             "'puregpu' for pure-GPU training.")
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train.')
    args = parser.parse_args()
    device = "cuda:7"
    batchs, num_features, num_classes = dataloader(mode="train", device=device)
    model = BlockGCN(in_feats=num_features,  #2708
            h_feats=128,  # 
            num_classes=num_classes).to(device)   
    loss_fn = flow.nn.CrossEntropyLoss().to(device)
    optimizer = flow.optim.Adam(model.parameters(),
                        lr=0.01, weight_decay=5e-4)
  
    
    for epoch in range(args.epochs):
        model.train()
        with tqdm.tqdm(batchs) as tq:
            for step, (input_nodes, output_nodes, blocks, adjs) in enumerate(tq):
                input_features = blocks[0]["src_feat"]
                output_labels = blocks[-1]["dst_label"]

                output_prediction = model(input_features, adjs) 

                loss = loss_fn(output_prediction, output_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                accuracy = sklearn.metrics.accuracy_score(output_labels.cpu().numpy(), output_prediction.argmax(1).detach().cpu().numpy())

                tq.set_postfix({'loss': '%.03f' % loss.item(), 'acc': '%.03f' % accuracy}, refresh=False)
  
    model.eval()

    valid_data, _ , _ = dataloader(mode="valid", device = device)
    predictions = []
    labels = []
    with tqdm.tqdm(valid_data) as tq, flow.no_grad():
        for input_nodes, output_nodes, blocks, adjs in tq:
            inputs = blocks[0]['src_feat']
            labels.append(blocks[-1]['dst_label'].cpu().numpy())
            predictions.append(model(inputs, adjs).argmax(1).cpu().numpy())
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        accuracy = sklearn.metrics.accuracy_score(labels, predictions)
        print('Epoch {} Validation Accuracy {}'.format(epoch, accuracy))