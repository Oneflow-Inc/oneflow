set -ux

DEVICE_NUM_PER_NODE=4
MASTER_ADDR=127.0.0.1
NUM_NODES=1
NODE_RANK=0

export ONEFLOW_DEBUG_MODE=1

python3 -m oneflow.distributed.launch \
    --nproc_per_node $DEVICE_NUM_PER_NODE \
    --nnodes $NUM_NODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    test_graph_pipeline.py
