# Everything related with GCN
branch: dev-gcn-spmm

## SpMM
The SpMM `Y=AX` is implemented on the basis of CuSparse. Two variants are given:

### spmm_coo
`spmm_coo` is firstly implemented because graphs are usually described in COO format.

Nevertheless, we found out that (1) the underlying cusparse kernels requires the COO sorted in row, which is not the case for many natural graph representation, and (2) Pytorch actually uses spmm_csr to perform spmm.

### spmm_csr
To mimic pytorch, the `spmm_csr` is implemented. To reduce the overhead of converting COO to CSR in every iteration of SpMM (as in Pytorch), the graphs are formatted to CSR during the data preparation.

## GCN
`gcn_spmm.py, train.py`

GCN layer is composed by two essential operations: dense-dense matrix multiplication and sparse-dense matrix multiplication. Based on the `MatMul` and `spmm_csr`, a 2-layer GCN can be easily constructed.

## Graph Sampling
`batchloader.py, batch_gcn.py`

Graph sampling is to make mini-batches for large-scale graph training. For a 2-layer GCN, each batch consists of two subgraphs, and each subgraph is processed by one single layer. The subgraph `{A}` is non-square (e.g., 1000 src nodes X 100 dst nodes). The sampling function is directly borrowed from DGL. 

For the full-batch training, the input of adjacency matrix `A` is a bidirected square, that is `A = A^T`. Thus, SpMM in forwarding (i.e., `Y = A X`) and backwarding (i.e., `dX = A^T dY`) directions can expressed using the same `spmm_csr` op without transposing the matrix `A`. However, for mini-batch training, `{A}` is not square anymore, and thus SpMM should be accordingly modified to enable the tranpose of `{A}` for backwarding propagation.

## GCN with OneEmbedding 
`oe_graph.py`

The input node features can be ported to OneEmbedding. To do so, we need to firstly write the features into the embedding in disk. It is not a clever way, but is the only way. GCN and an embedding layer (containing {id : feature} pairs) constitue a GCN module for training. The input should be vertex ids, which is used to locate the features.

## TODO: combing OneEmbedding with Graph Sampling
Graph sampling make OneEmbedding difficult. The mini-batch GCN take two subgraphs and corresponding features as inputs. The subgraphs are represented in CSR and relabelled with new ids. We need to find a way to map the new ids and their original ids in the embedding. 