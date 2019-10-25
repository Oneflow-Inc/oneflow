# !/usr/bin/env python

import os
import sys
import time
import argparse
from datetime import datetime

_DATA_DIR = '/dataset/bert/of_wiki_seq_len_128'
_MODEL_LOAD = "/dataset/model_zoo/bert_new_snapshot/of_L-12_H-768_A-12_random_init"
_MODEL_SAVE_DIR = "./model_save-{}".format(
    str(datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
)
NODE_LIST = "192.168.1.15,192.168.1.16"
parser = argparse.ArgumentParser(description="flags for bert")

# resouce
parser.add_argument("--gpu_num_per_node", type=int, default=1)
parser.add_argument("--node_num", type=int, default=1)
parser.add_argument("--node_list", type=str, default=NODE_LIST)

# train
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--weight_l2", type=float, default=0.01, help="weight l2 decay parameter")
parser.add_argument("--batch_size_per_device", type=int, default=24)
parser.add_argument("--iter_num", type=int, default=10, help="total iterations to run")
parser.add_argument("--log_every_n_iter", type=int, default=1, help="print loss every n iteration")
parser.add_argument("--data_dir", type=str, default=_DATA_DIR)
parser.add_argument("--data_part_num", type=int, default=32, help="data part number in dataset")
parser.add_argument("--model_load_dir", type=str, default=_MODEL_LOAD)
parser.add_argument("--model_save_dir", type=str, default=_MODEL_SAVE_DIR)

# bert
parser.add_argument("--seq_length", type=int, default=512)
parser.add_argument("--max_predictions_per_seq", type=int, default=80)
parser.add_argument("--num_hidden_layers", type=int, default=24)
parser.add_argument("--num_attention_heads", type=int, default=16)
parser.add_argument("--max_position_embeddings", type=int, default=512)
parser.add_argument("--type_vocab_size", type=int, default=2)
parser.add_argument("--vocab_size", type=int, default=30522)
parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.1)
parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
parser.add_argument("--hidden_size_per_head", type=int, default=64)
parser.add_argument("--enable_auto_mixed_precision", type=bool, default=False)

args = parser.parse_args()

