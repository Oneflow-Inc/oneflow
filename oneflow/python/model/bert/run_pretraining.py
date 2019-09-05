import os
import sys
import time
import argparse
import numpy as np
from datetime import datetime

import oneflow as flow
from pretrain import PreTrain#, Eval

_DATA_DIR = '/dataset/bert/of_wiki_seq_len_128'
#_DATA_DIR = '/dataset/bert/bert_seq_len_128_repeat1024'
#_DATA_DIR = '/dataset/bert_regression_test/0'
#_MODEL_LOAD = "/dataset/model_zoo/bert/of_L-12_H-768_A-12_random_init"
_MODEL_LOAD = "/dataset/model_zoo/bert_new_snapshot/of_L-12_H-768_A-12_random_init"

parser = argparse.ArgumentParser(description="flags for bert")
parser.add_argument("-d", "--device_num_per_node", type=int, default=4)
parser.add_argument("-n", "--node_num", type=int, default=1)
parser.add_argument("-b", "--batch_size_per_device", type=int, default=24)
parser.add_argument("-s", "--num_steps", type=int, default=100)
parser.add_argument("-c", "--copy_binary_to_worker", type=bool, default=True)
parser.add_argument("-u", "--use_uuid", type=bool, default=False)
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

  decoders = BertDecoder(_DATA_DIR, seq_length, max_predictions_per_seq)

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

def PretrainJob():
    total_device_num = args.node_num * args.device_num_per_node
    batch_size = total_device_num * args.batch_size_per_device
    data_part_num = total_device_num #use total_device_num for test

    job_conf = flow.get_cur_job_conf_builder()
    job_conf.batch_size(batch_size).data_part_num(data_part_num).default_data_type(flow.float)
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

  start_time = time.time()
  config = flow.ConfigProtoBuilder()
  config.gpu_device_num(args.device_num_per_node)
  config.grpc_use_no_signal()
  config.ctrl_port(9917)
  config.data_port(9927)
  config.machine(nodes[:args.node_num])

  assert args.node_num <= len(nodes)
  if args.node_num > 1:
    flow.deprecated.init_worker(config, scp_binary=args.copy_binary_to_worker, use_uuid=args.use_uuid)
  flow.init(config)

  flow.add_job(PretrainJob)
  with flow.Session() as sess:
    check_point = flow.train.SimpleCheckPointManager('model_save')
    check_point.initialize_or_restore()
    #check_point = flow.train.CheckPoint()
    #check_point.load(_MODEL_LOAD)
    fmt_str = "{:>12}  {:>12}  {:>12.10f}"
    print('{:>12}  {:14}  {}'.format( "step", "loss", "time"))
    train_start_time = time.time()
    step_time = []
    for i in range(args.num_steps):
      loss_mean = sess.run(PretrainJob).get().mean()
      step_time.append(time.time())
      train_step_time = step_time[i] - step_time[i-1]
      print(fmt_str.format(i, loss_mean, train_step_time))
      #sess.no_return_run(PretrainJob)#.async_get(AsyncGetCallback)
      #sess.run(PretrainJob).async_get(AsyncGetCallback)
    check_point.save()
  total_time = step_time[-1] - start_time
  train_time = step_time[-1] - train_start_time
  initial_time = train_start_time - start_time
  mean_batch_time = (step_time[-1] - step_time[0]) / (args.num_steps - 1)
  total_batch_size = args.node_num * args.device_num_per_node * args.batch_size_per_device
  throughput = total_batch_size / mean_batch_time

  print('total time', total_time)
  print('initial time', initial_time)
  print('first loss time', step_time[0] - start_time) #include model init and first batch cal time.
  print('train time', train_time)
  print('last - first loss time', step_time[-1] - step_time[0])
  print('average batch time', mean_batch_time)
  print('samples/sec', throughput)
  print('destroy time', time.time() - step_time[-1])
