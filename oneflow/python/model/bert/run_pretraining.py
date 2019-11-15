import os
import sys
import time
import argparse
import shutil
import numpy as np
from datetime import datetime

import oneflow as flow
from pretrain import PreTrain#, Eval

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

args = parser.parse_args()

def _blob_conf(name, shape, dtype=flow.int32):
  return flow.data.BlobConf(name=name, shape=shape, dtype=dtype, codec=flow.data.RawCodec())

def BertDecoder(data_dir, batch_size=1, data_part_num=1, seq_length=128, max_predictions_per_seq=20):
  blob_confs = []
  blob_confs.append(_blob_conf('input_ids', [seq_length]))
  blob_confs.append(_blob_conf('next_sentence_labels', [1]))
  blob_confs.append(_blob_conf('input_mask', [seq_length]))
  blob_confs.append(_blob_conf('segment_ids', [seq_length]))
  blob_confs.append(_blob_conf('masked_lm_ids', [max_predictions_per_seq]))
  blob_confs.append(_blob_conf('masked_lm_positions', [max_predictions_per_seq]))
  blob_confs.append(_blob_conf('masked_lm_weights', [max_predictions_per_seq], flow.float))
  return flow.data.decode_ofrecord(data_dir, blob_confs,
                                   batch_size=batch_size,
                                   name="decode",
                                   data_part_num=data_part_num)

def BuildPreTrainNet(batch_size, data_part_num, seq_length=128, max_position_embeddings=512,
                     num_hidden_layers=12, num_attention_heads=12,
                     hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
                     vocab_size=30522, type_vocab_size=2, max_predictions_per_seq=20):

  hidden_size = 64 * num_attention_heads#, H = 64, size per head
  intermediate_size = hidden_size * 4

  decoders = BertDecoder(args.data_dir, batch_size, data_part_num, seq_length,
                         max_predictions_per_seq)

  input_ids = decoders[0]
  next_sentence_labels = decoders[1]
  input_mask = decoders[2]
  token_type_ids = decoders[3]
  masked_lm_ids = decoders[4]
  masked_lm_positions = decoders[5]
  masked_lm_weights = decoders[6]
  return PreTrain(input_ids,
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
                  initializer_range=0.02)

_BERT_MODEL_UPDATE_CONF = dict(
  learning_rate_decay = dict(
    polynomial_conf = dict(
      decay_batches = 100000,
      end_learning_rate = 0.0,
    )
  ),
  warmup_conf = dict(
    linear_conf = dict(
      warmup_batches = 1000,
      start_multiplier = 0,
    )
  ),
  clip_conf = dict(
    clip_by_global_norm = dict(
      clip_norm = 1.0,
    )
  ),
  adam_conf = dict(
    epsilon = 1e-6
  ),
)

@flow.function
def PretrainJob():
  total_device_num = args.node_num * args.gpu_num_per_node
  batch_size = total_device_num * args.batch_size_per_device

  flow.config.train.primary_lr(args.learning_rate)
  flow.config.train.model_update_conf(_BERT_MODEL_UPDATE_CONF)
  flow.config.train.weight_l2(args.weight_l2)

  loss = BuildPreTrainNet(batch_size, args.data_part_num,
                          seq_length=args.seq_length,
                          max_position_embeddings=args.max_position_embeddings,
                          num_hidden_layers=args.num_hidden_layers,
                          num_attention_heads=args.num_attention_heads,
                          hidden_dropout_prob=args.hidden_dropout_prob,
                          attention_probs_dropout_prob=args.attention_probs_dropout_prob,
                          vocab_size=args.vocab_size,
                          type_vocab_size=args.type_vocab_size,
                          max_predictions_per_seq=args.max_predictions_per_seq)
  flow.losses.add_loss(loss)
  return loss

cur_step = 0
def AsyncGetCallback(result):
  global cur_step
  print('{:>12}  {:>.10f}  {:.2f}'.format(cur_step, result.mean(), time.time()))
  cur_step += 1

if __name__ == '__main__':
  for arg in vars(args):
    print('{} = {}'.format(arg, getattr(args, arg)))

  start_time = time.time()
  flow.config.gpu_device_num(args.gpu_num_per_node)
  flow.config.ctrl_port(9788)
  flow.config.data_port(9789)
  flow.config.default_data_type(flow.float)
  flow.config.enable_inplace(False)

  if args.node_num > 1:
    flow.config.ctrl_port(12138)
    nodes = []
    for n in args.node_list.strip().split(","):
      addr_dict = {}
      addr_dict["addr"] = n
      nodes.append(addr_dict)

    flow.config.machine(nodes)

  check_point = flow.train.CheckPoint()
  if args.model_load_dir != '':
    assert os.path.isdir(args.model_load_dir)
    check_point.load(args.model_load_dir)
    print('init model from {}'.format(args.model_load_dir))
  else:
    check_point.init()
    print('init model on demand')

  fmt_str = "{:>12}  {:>12}  {:>12.10f}"
  print('{:>12}  {:14}  {}'.format( "step", "loss", "time"))
  train_start_time = time.time()
  step_time = []
  for step in range(args.iter_num):
    if args.model_save_dir != '':
      if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
      assert args.log_every_n_iter > 0
      if step % args.log_every_n_iter == 0:
        snapshot_save_path = os.path.join(args.model_save_dir, 'snapshot_%d'%(step))
        check_point.save(snapshot_save_path)

    loss_mean = PretrainJob().get().mean()
    step_time.append(time.time())
    train_step_time = step_time[step] - step_time[step-1]
    print(fmt_str.format(step, loss_mean, train_step_time))
  snapshot_save_path = os.path.join(args.model_save_dir, 'last_snapshot')
  check_point.save(snapshot_save_path)


  total_time = step_time[-1] - start_time
  train_time = step_time[-1] - train_start_time
  init_time = train_start_time - start_time
  mean_batch_time = (step_time[-1] - step_time[0]) / (args.iter_num - 1)
  total_batch_size = args.node_num * args.gpu_num_per_node * args.batch_size_per_device
  throughput = total_batch_size / mean_batch_time

  print('total time', total_time)
  print('init time', init_time)
  print('first loss time', step_time[0] - start_time) #include model init and first batch cal time.
  print('train time', train_time)
  print('last - first loss time', step_time[-1] - step_time[0])
  print('average batch time', mean_batch_time)
  print('samples/sec', throughput)
  print('destroy time', time.time() - step_time[-1])
