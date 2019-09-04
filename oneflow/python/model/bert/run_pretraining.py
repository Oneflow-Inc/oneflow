import os
import sys
import time
import argparse
import numpy as np
from datetime import datetime

import oneflow as flow
from pretrain import PreTrain#, Eval

#_DATA_DIR = '/dataset/bert/of_wiki_seq_len_128'
_DATA_DIR = '/dataset/bert/bert_seq_len_128_repeat1024'
#_DATA_DIR = '/dataset/bert_regression_test/0'
#_MODEL_LOAD = "/dataset/model_zoo/bert/of_L-12_H-768_A-12_random_init"
#_MODEL_LOAD = "/dataset/model_zoo/bert_new_snapshot/of_L-12_H-768_A-12_random_init"
_MODEL_LOAD = "/home/xiexuan/work/bert_job_set/snapshots/snapshot_2019_08_31_17_32_38_663"
_MODEL_SAVE = './snapshots'

parser = argparse.ArgumentParser(description="flags for multi-node and resource")
parser.add_argument("-d", "--device_num_per_node", type=int, default=1)
parser.add_argument("-n", "--node_num", type=int, default=1)
parser.add_argument("-b", "--batch_size_per_device", type=int, default=8)
parser.add_argument("-s", "--num_steps", type=int, default=100)
args = parser.parse_args()

nodes = [{'addr':'192.168.1.16'},{'addr':'192.168.1.15'}]

def _blob_conf(name, shape, dtype=flow.int32):
  return flow.data.BlobConf(name=name, shape=shape, dtype=dtype, codec=flow.data.RawCodec())

def BertDecoder(data_dir='', seq_length=128, max_predictions_per_seq=20):
  blob_confs = []
  blob_confs.append(_blob_conf('input_ids', [seq_length]))
  blob_confs.append(_blob_conf('next_sentence_labels', [1]))
  blob_confs.append(_blob_conf('input_mask', [seq_length]))
  blob_confs.append(_blob_conf('segment_ids', [seq_length]))
  blob_confs.append(_blob_conf('masked_lm_ids', [max_predictions_per_seq]))
  blob_confs.append(_blob_conf('masked_lm_positions', [max_predictions_per_seq]))
  blob_confs.append(_blob_conf('masked_lm_weights', [max_predictions_per_seq], flow.float))
  return flow.data.decode_ofrecord(data_dir, blob_confs, name="decode")

def BuildPreTrainNet(seq_length=128, max_position_embeddings=512, num_hidden_layers=12,
                     num_attention_heads=12, hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
                     vocab_size=30522, type_vocab_size=2, max_predictions_per_seq=20):

  hidden_size = 64 * num_attention_heads#, H = 64, size per head
  intermediate_size = hidden_size * 4

  with flow.deprecated.variable_scope('other'):
    decoders = BertDecoder(_DATA_DIR, seq_length, max_predictions_per_seq)

  # input blobs
  #input_ids = decoders['input_ids']
  #next_sentence_labels = decoders['next_sentence_labels']
  #token_type_ids = decoders['segment_ids']
  #input_mask = decoders['input_mask']
  #masked_lm_ids = decoders['masked_lm_ids']
  #masked_lm_positions = decoders['masked_lm_positions']
  #masked_lm_weights = decoders['masked_lm_weights']

  input_ids = decoders[0]
  next_sentence_labels = decoders[1]
  token_type_ids = decoders[2]
  input_mask = decoders[3]
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
  #warmup_conf = dict(
  #  linear_conf = dict(
  #    warmup_batches = 1000,
  #    start_multiplier = 0,
  #  )
  #),
  clip_conf = dict(
    clip_by_global_norm = dict(
      clip_norm = 1.0,
    )
  ),
  #momentum_conf = dict(
  #lars_conf = dict(
  adam_conf = dict(
    epsilon = 1e-6
  ),
)

def PretrainJob():
    total_device_num = args.node_num * args.device_num_per_node
    batch_size = total_device_num * args.batch_size_per_device

    job_conf = flow.get_cur_job_conf_builder()
    job_conf.batch_size(batch_size).data_part_num(total_device_num).default_data_type(flow.float)
    job_conf.default_initializer_conf(dict(constant_conf=dict(value=0.0)))
    #job_conf.enable_nccl(False)
    #job_conf.enable_cuda_ring_all_reduce()
    job_conf.train_conf()
    job_conf.train_conf().batch_size = batch_size
    job_conf.train_conf().primary_lr = 1e-4
    job_conf.train_conf().weight_l2 = 0.01
    job_conf.train_conf().num_of_batches_in_snapshot = 1000
    job_conf.model_update_conf(_BERT_MODEL_UPDATE_CONF)

    job_conf.enable_inplace(False)
    loss = BuildPreTrainNet(hidden_dropout_prob=0, attention_probs_dropout_prob=0)
    flow.losses.add_loss(loss)
    return loss

cur_step = 0
def AsyncGetCallback(result):
  global cur_step
  print('{:>12}  {:>.10f}  {:.2f}'.format(cur_step, result.mean(), time.time()))
  cur_step += 1

if __name__ == '__main__':
  print('node/machine num', args.node_num)
  print('device/gpu num per node/machine', args.device_num_per_node)
  print('batch size per device/gpu', args.batch_size_per_device)
  print('number of steps', args.num_steps)

  config = flow.ConfigProtoBuilder()
  config.gpu_device_num(args.device_num_per_node)
  config.grpc_use_no_signal()
  config.ctrl_port(9917)
  config.data_port(9927)
  config.machine(nodes[:args.node_num])
  #config.model_load_snapshot_path(_MODEL_LOAD)
  #config.model_save_snapshots_path(_MODEL_SAVE)

  assert args.node_num == 1, 'support 1 node currently'
  flow.init(config)

  flow.add_job(PretrainJob)
  with flow.Session() as sess:
    check_point = flow.train.SimpleCheckPointManager('mode_save')
    check_point.initialize_or_restore()
    # sess.sync()
    print('{:>12}  {:14}  {}'.format( "step", "loss", "time"))
    for i in range(args.num_steps):
      print(fmt_str.format(i, "train loss:", sess.run(PretrainJob).get().mean()))
      #sess.no_return_run(PretrainJob)#.async_get(AsyncGetCallback)
      #sess.run(PretrainJob).async_get(AsyncGetCallback)
    check_point.save(session=sess)

