"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import absolute_import, division, print_function

import argparse
import os
import random
import time
from collections import OrderedDict
from datetime import datetime

import benchmark_util
import oneflow as flow
from pretrain import PreTrain

parser = argparse.ArgumentParser(description="flags for bert")

# resouce
parser.add_argument("--gpu_num_per_node", type=int, default=1)
parser.add_argument("--node_num", type=int, default=1)
parser.add_argument("--node_list", type=str, default=None)

# train
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
parser.add_argument(
    "--weight_decay_rate", type=float, default=0.01, help="weight decay rate"
)
parser.add_argument("--batch_size_per_device", type=int, default=24)
parser.add_argument("--iter_num", type=int, default=10, help="total iterations to run")
parser.add_argument(
    "--skip_iter_num",
    type=int,
    default=10,
    help="number of skipping iterations for benchmark purpose.",
)
parser.add_argument(
    "--log_every_n_iter", type=int, default=1, help="print loss every n iteration"
)
parser.add_argument("--data_dir", type=str, default=None)
parser.add_argument(
    "--data_part_num", type=int, default=32, help="data part number in dataset"
)
parser.add_argument(
    "--enable_auto_mixed_precision",
    default=False,
    type=lambda x: (str(x).lower() == "true"),
)

# log and resore/save
parser.add_argument(
    "--loss_print_every_n_iter",
    type=int,
    default=1,
    required=False,
    help="print loss every n iteration",
)
parser.add_argument(
    "--model_save_every_n_iter",
    type=int,
    default=200,
    required=False,
    help="save model every n iteration",
)
parser.add_argument(
    "--model_save_dir",
    type=str,
    default="./output/model_save-{}".format(
        str(datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
    ),
    required=False,
    help="model save directory",
)
parser.add_argument(
    "--save_last_snapshot",
    type=bool,
    default=False,
    required=False,
    help="save model snapshot for last iteration",
)
parser.add_argument(
    "--model_load_dir",
    type=str,
    default=None,
    required=False,
    help="model load directory",
)
parser.add_argument(
    "--log_dir",
    type=str,
    default="./output",
    required=False,
    help="log info save directory",
)

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

parser.add_argument("--warmup_batches", type=int, default=1000)
parser.add_argument("--lr_decay_num", type=int, default=100000)
parser.add_argument(
    "--lr_decay_num_same_as_iter_num",
    default=False,
    type=(lambda x: str(x).lower() == "true"),
)

args = parser.parse_args()


def _blob_conf(name, shape, dtype=flow.int32):
    return flow.data.BlobConf(
        name=name, shape=shape, dtype=dtype, codec=flow.data.RawCodec()
    )


def BertDecoder(
    data_dir, batch_size, data_part_num, seq_length, max_predictions_per_seq
):
    config_ordered_dict = OrderedDict()
    config_ordered_dict["input_ids"] = seq_length
    config_ordered_dict["next_sentence_labels"] = 1
    config_ordered_dict["input_mask"] = seq_length
    config_ordered_dict["segment_ids"] = seq_length
    config_ordered_dict["masked_lm_ids"] = max_predictions_per_seq
    config_ordered_dict["masked_lm_positions"] = max_predictions_per_seq
    config_ordered_dict["masked_lm_weights"] = max_predictions_per_seq

    ofrecord = flow.data.ofrecord_reader(
        data_dir, batch_size=batch_size, data_part_num=data_part_num, name="decode",
    )
    ret = {}
    for k, v in config_ordered_dict.items():
        ret[k] = flow.data.ofrecord_raw_decoder(
            ofrecord,
            k,
            shape=(v,),
            dtype=flow.float if k == "masked_lm_weights" else flow.int32,
        )
    return ret


def BuildPreTrainNet(
    batch_size,
    data_part_num,
    seq_length=128,
    max_position_embeddings=512,
    num_hidden_layers=12,
    num_attention_heads=12,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    vocab_size=30522,
    type_vocab_size=2,
    max_predictions_per_seq=20,
):
    hidden_size = 64 * num_attention_heads  # , H = 64, size per head
    intermediate_size = hidden_size * 4

    decoders = BertDecoder(
        args.data_dir, batch_size, data_part_num, seq_length, max_predictions_per_seq
    )

    input_ids = decoders["input_ids"]
    next_sentence_labels = decoders["next_sentence_labels"]
    input_mask = decoders["input_mask"]
    token_type_ids = decoders["segment_ids"]  # note: segment_ids = token_type_ids
    masked_lm_ids = decoders["masked_lm_ids"]
    masked_lm_positions = decoders["masked_lm_positions"]
    masked_lm_weights = decoders["masked_lm_weights"]
    return PreTrain(
        input_ids,
        input_mask,
        token_type_ids,
        masked_lm_positions,
        masked_lm_ids,
        masked_lm_weights,
        next_sentence_labels,
        vocab_size,
        seq_length=seq_length,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        hidden_act="gelu",
        hidden_dropout_prob=hidden_dropout_prob,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        max_position_embeddings=max_position_embeddings,
        type_vocab_size=type_vocab_size,
        max_predictions_per_seq=max_predictions_per_seq,
        initializer_range=0.02,
    )


_BERT_MODEL_UPDATE_CONF = dict(
    learning_rate_decay=dict(
        polynomial_conf=dict(
            decay_batches=args.iter_num
            if args.lr_decay_num_same_as_iter_num
            else args.lr_decay_num,
            end_learning_rate=0.0,
        )
    ),
    warmup_conf=dict(
        linear_conf=dict(warmup_batches=args.warmup_batches, start_multiplier=0,)
    ),
    clip_conf=dict(clip_by_global_norm=dict(clip_norm=1.0,)),
    adam_conf=dict(epsilon=1e-6),
    weight_decay_conf=dict(
        weight_decay_rate=args.weight_decay_rate,
        excludes=dict(pattern=["bias", "LayerNorm", "layer_norm"]),
    ),
)

func_config = flow.FunctionConfig()
func_config.default_distribute_strategy(flow.scope.consistent_view())
func_config.train.primary_lr(args.learning_rate)
func_config.default_data_type(flow.float)
func_config.train.model_update_conf(_BERT_MODEL_UPDATE_CONF)
func_config.enable_auto_mixed_precision(args.enable_auto_mixed_precision)


flow.config.gpu_device_num(args.gpu_num_per_node)


@flow.global_function(func_config)
def PretrainJob():
    total_device_num = args.node_num * args.gpu_num_per_node
    batch_size = total_device_num * args.batch_size_per_device

    total_loss, mlm_loss, nsp_loss = BuildPreTrainNet(
        batch_size,
        args.data_part_num,
        seq_length=args.seq_length,
        max_position_embeddings=args.max_position_embeddings,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        hidden_dropout_prob=args.hidden_dropout_prob,
        attention_probs_dropout_prob=args.attention_probs_dropout_prob,
        vocab_size=args.vocab_size,
        type_vocab_size=args.type_vocab_size,
        max_predictions_per_seq=args.max_predictions_per_seq,
    )
    flow.losses.add_loss(total_loss)
    return total_loss, mlm_loss, nsp_loss


def main():
    print("=".ljust(66, "="))
    print(
        "Running bert: num_gpu_per_node = {}, num_nodes = {}.".format(
            args.gpu_num_per_node, args.node_num
        )
    )
    print("=".ljust(66, "="))
    for arg in vars(args):
        print("{} = {}".format(arg, getattr(args, arg)))
    print("-".ljust(66, "-"))
    print("Time stamp: {}".format(str(datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))))

    flow.env.log_dir(args.log_dir)

    if args.node_num > 1:
        nodes = []
        for n in args.node_list.strip().split(","):
            addr_dict = {}
            addr_dict["addr"] = n
            nodes.append(addr_dict)

        flow.env.machine(nodes)
    if os.getenv("ONEFLOW_DEBUG_MODE"):
        flow.env.ctrl_port(9788)
    check_point = flow.train.CheckPoint()
    if args.model_load_dir:
        assert os.path.isdir(args.model_load_dir)
        check_point.load(args.model_load_dir)
        print("Restoring model from {}.".format(args.model_load_dir))
    else:
        check_point.init()
        print("Init model on demand")

    total_batch_size = (
        args.node_num * args.gpu_num_per_node * args.batch_size_per_device
    )
    speedometer = benchmark_util.BERTSpeedometer()
    start_time = time.time()

    for step in range(args.skip_iter_num + args.iter_num):
        cb = speedometer.speedometer_cb(
            step,
            start_time,
            total_batch_size,
            args.skip_iter_num,
            args.iter_num,
            args.loss_print_every_n_iter,
        )
        PretrainJob().async_get(cb)

        if (step + 1) % args.model_save_every_n_iter == 0:
            if not os.path.exists(args.model_save_dir):
                os.makedirs(args.model_save_dir)
            snapshot_save_path = os.path.join(
                args.model_save_dir, "snapshot_%d" % (step + 1)
            )
            print("Saving model to {}.".format(snapshot_save_path))
            check_point.save(snapshot_save_path)

    if args.save_last_snapshot:
        snapshot_save_path = os.path.join(args.model_save_dir, "last_snapshot")
        if not os.path.exists(snapshot_save_path):
            os.makedirs(snapshot_save_path)
        print("Saving model to {}.".format(snapshot_save_path))
        check_point.save(snapshot_save_path)


if __name__ == "__main__":
    main()
