import os
from datetime import datetime
import time

import oneflow as flow
from pretrain import PreTrain#, Eval

_DATA_DIR = '/dataset/bert/of_wiki_seq_len_128'
#_DATA_DIR = '/dataset/bert_regression_test/0'
_MODEL_LOAD = "/dataset/model_zoo/bert/of_L-12_H-768_A-12_random_init"
_MODEL_SAVE = './model_save-{}'.format(str(datetime.now().strftime('%Y-%m-%d-%H:%M:%S')))

def BertDecoder(dl_net, data_dir='', seq_length=128, max_predictions_per_seq=20):
  return dl_net.DecodeOFRecord(data_dir, name='decode', blob=[
    {
      'name': 'input_ids',
      'shape': {'dim': [seq_length]},
      'data_type': flow.int32,
      'encode_case': {'raw': {}},
    },
    {
      'name': 'next_sentence_labels',
      'shape': {'dim': [1]},
      'data_type': flow.int32,
      'encode_case': {'raw': {}},
    },
    {
      'name': 'input_mask',
      'shape': {'dim': [seq_length]},
      'data_type': flow.int32,
      'encode_case': {'raw': {}},
    },
    {
      'name': 'segment_ids',
      'shape': {'dim': [seq_length]},
      'data_type': flow.int32,
      'encode_case': {'raw': {}},
    },
    {
      'name': 'masked_lm_ids',
      'shape': {'dim': [max_predictions_per_seq]},
      'data_type': flow.int32,
      'encode_case': {'raw': {}},
    },
    {
      'name': 'masked_lm_positions',
      'shape': {'dim': [max_predictions_per_seq]},
      'data_type': flow.int32,
      'encode_case': {'raw': {}},
    },
    {
      'name': 'masked_lm_weights',
      'shape': {'dim': [max_predictions_per_seq]},
      'data_type': flow.float,
      'encode_case': {'raw': {}},
    }
  ]);

def BuildPreTrainNetWithDeprecatedAPI(seq_length=128, max_position_embeddings=512,
                                      num_hidden_layers=12, num_attention_heads=12,
                                      hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
                                      vocab_size=30522, type_vocab_size=2,
                                      max_predictions_per_seq=20):

  hidden_size = 64 * num_attention_heads#, H = 64, size per head
  intermediate_size = hidden_size * 4

  dl_net = flow.deprecated.get_cur_job_dlnet_builder()

  with dl_net.VariableScope('other'):
    decoders = BertDecoder(dl_net, _DATA_DIR, seq_length, max_predictions_per_seq)

  # input blobs
  input_ids = decoders['input_ids']
  next_sentence_labels = decoders['next_sentence_labels']
  token_type_ids = decoders['segment_ids']
  input_mask = decoders['input_mask']
  masked_lm_ids = decoders['masked_lm_ids']
  masked_lm_positions = decoders['masked_lm_positions']
  masked_lm_weights = decoders['masked_lm_weights']

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
    job_conf = flow.get_cur_job_conf_builder()
    job_conf.batch_size(96).data_part_num(4).default_data_type(flow.float)
    job_conf.default_initializer_conf(dict(constant_conf=dict(value=0.0)))
    job_conf.train_conf()
    job_conf.train_conf().primary_lr = 1e-4
    job_conf.train_conf().weight_l2 = 0.01
    job_conf.train_conf().num_of_batches_in_snapshot = 1000
    job_conf.model_update_conf(_BERT_MODEL_UPDATE_CONF)
    job_conf.train_conf().loss_lbn.extend(["identity_loss/loss"])
    job_conf.enable_inplace(False)
    return BuildPreTrainNetWithDeprecatedAPI(hidden_dropout_prob=0, attention_probs_dropout_prob=0)

cur_step = 0
def AsyncGetCallback(result):
  global cur_step
  print('{:>12}  {:>.4f}  {:.2f}'.format(cur_step, result.mean(), time.time()))
  cur_step += 1

if __name__ == '__main__':
  config = flow.ConfigProtoBuilder()
  config.ctrl_port(12137)
  config.machine([{'addr':'192.168.1.11'},{'addr':'192.168.1.15'}])
  config.gpu_device_num(2)
  config.grpc_use_no_signal()
  #config.model_load_snapshot_path(_MODEL_LOAD)
  config.model_save_snapshots_path(_MODEL_SAVE)
  flow.deprecated.init_worker_and_master(config)

  flow.add_job(PretrainJob)
  with flow.Session() as sess:
    check_point = flow.train.CheckPoint()
    check_point.restore().initialize_or_restore(session=sess)
    # sess.sync()
    print('{:>12}  {:8}  {}'.format( "step", "loss", "time"))
    for i in range(200):
      #print(fmt_str.format(i, "train loss:", sess.run(PretrainJob).get().mean()))
      #sess.no_return_run(PretrainJob)#.async_get(AsyncGetCallback)
      sess.run(PretrainJob).async_get(AsyncGetCallback)
  flow.deprecated.delete_worker(config)
