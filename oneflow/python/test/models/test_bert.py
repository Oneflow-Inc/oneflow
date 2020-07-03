import copy
import sys

import numpy as np
import oneflow as flow
from absl import flags
from pretrain import PreTrain

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", "/dataset/bert/bert_seq_len_128_repeat1024", "")
flags.DEFINE_string(
    "model_load_dir", "/dataset/bert_regression_test/of_random_init_L-12_H-768_A-12", ""
)
flags.DEFINE_string("model_save_dir", "snapshots", "")
flags.DEFINE_float("lr", 1e-4, "learning rate")
flags.DEFINE_float("weight_decay_rate", 0.01, "")
flags.DEFINE_integer("batch_size", 24, "")
flags.DEFINE_integer("data_part_num", 8, "")
flags.DEFINE_integer("seq_length", 128, "")
flags.DEFINE_integer("max_predictions_per_seq", 20, "")
flags.DEFINE_integer("num_hidden_layers", 12, "")
flags.DEFINE_integer("num_attention_heads", 12, "")
flags.DEFINE_integer("max_position_embeddings", 512, "")
flags.DEFINE_integer("type_vocab_size", 2, "")
flags.DEFINE_integer("vocab_size", 30522, "")
flags.DEFINE_float("attention_probs_dropout_prob", 0.0, "")
flags.DEFINE_float("hidden_dropout_prob", 0.0, "")
flags.DEFINE_integer("hidden_size_per_head", 64, "")
FLAGS(sys.argv)


def _blob_conf(name, shape, dtype=flow.int32):
    return flow.data.BlobConf(
        name=name, shape=shape, dtype=dtype, codec=flow.data.RawCodec()
    )


def BertDecoder(
    data_dir, batch_size=1, data_part_num=1, seq_length=128, max_predictions_per_seq=20
):
    blob_confs = []
    blob_confs.append(_blob_conf("input_ids", [seq_length]))
    blob_confs.append(_blob_conf("next_sentence_labels", [1]))
    blob_confs.append(_blob_conf("input_mask", [seq_length]))
    blob_confs.append(_blob_conf("segment_ids", [seq_length]))
    blob_confs.append(_blob_conf("masked_lm_ids", [max_predictions_per_seq]))
    blob_confs.append(_blob_conf("masked_lm_positions", [max_predictions_per_seq]))
    blob_confs.append(
        _blob_conf("masked_lm_weights", [max_predictions_per_seq], flow.float)
    )
    return flow.data.decode_ofrecord(
        data_dir,
        blob_confs,
        batch_size=batch_size,
        name="decode",
        data_part_num=data_part_num,
    )


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

    hidden_size = 64 * num_attention_heads
    intermediate_size = hidden_size * 4

    decoders = BertDecoder(
        FLAGS.data_dir, batch_size, data_part_num, seq_length, max_predictions_per_seq
    )

    input_ids = decoders[0]
    next_sentence_labels = decoders[1]
    input_mask = decoders[2]
    token_type_ids = decoders[3]
    masked_lm_ids = decoders[4]
    masked_lm_positions = decoders[5]
    masked_lm_weights = decoders[6]
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
        polynomial_conf=dict(decay_batches=100000, end_learning_rate=0.0)
    ),
    warmup_conf=dict(linear_conf=dict(warmup_batches=1000, start_multiplier=0)),
    clip_conf=dict(clip_by_global_norm=dict(clip_norm=1.0)),
    adam_conf=dict(epsilon=1e-6),
    weight_decay_conf=dict(
        weight_decay_rate=FLAGS.weight_decay_rate,
        excludes=dict(pattern=["bias", "LayerNorm", "layer_norm"]),
    ),
)


def PretrainJob():
    loss = BuildPreTrainNet(
        batch_size=FLAGS.batch_size,
        data_part_num=FLAGS.data_part_num,
        seq_length=FLAGS.seq_length,
        max_position_embeddings=FLAGS.max_position_embeddings,
        num_hidden_layers=FLAGS.num_hidden_layers,
        num_attention_heads=FLAGS.num_attention_heads,
        hidden_dropout_prob=FLAGS.hidden_dropout_prob,
        attention_probs_dropout_prob=FLAGS.attention_probs_dropout_prob,
        vocab_size=FLAGS.vocab_size,
        type_vocab_size=FLAGS.type_vocab_size,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
    )
    flow.losses.add_loss(loss)
    return loss


func_config = flow.FunctionConfig()
func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
func_config.train.primary_lr(FLAGS.lr)
func_config.train.model_update_conf(_BERT_MODEL_UPDATE_CONF)
func_config.enable_auto_mixed_precision(FLAGS.enable_auto_mixed_precision)


def test_1n1c(test_case):
    flow.config.enable_debug_mode(True)
    flow.config.gpu_device_num(1)
    pretrain_job = flow.global_function(func_config)(PretrainJob)
    check_point = flow.train.CheckPoint()
    check_point.load(FLAGS.model_load_dir)
    of_loss = [pretrain_job().get().mean() for _ in range(10)]
    print(of_loss)


def test_1n4c(test_case):
    flow.config.gpu_device_num(4)
    pretrain_job = flow.global_function(func_config)(PretrainJob)
    check_point = flow.train.CheckPoint()
    check_point.load(FLAGS.model_load_dir)
    of_loss = [pretrain_job().get().mean() for _ in range(10)]
    print(of_loss)


@flow.unittest.num_nodes_required(2)
def test_2n8c(test_case):
    flow.config.gpu_device_num(4)
    pretrain_job = flow.global_function(func_config)(PretrainJob)
    check_point = flow.train.CheckPoint()
    check_point.load(FLAGS.model_load_dir)
    of_loss = [pretrain_job().get().mean() for _ in range(10)]
    print(of_loss)


def test_inplace(test_case):
    test_case.assertTrue(
        np.allclose(GetSeveralLossesAsNumpy(True), GetSeveralLossesAsNumpy(False))
    )


def GetSeveralLossesAsNumpy(enable_inplace, num_iters=10):
    flow.config.enable_debug_mode(True)
    flow.config.gpu_device_num(1)
    train_config = flow.FunctionConfig()
    train_config.default_distribute_strategy(flow.distribute.consistent_strategy())
    train_config.train.primary_lr(FLAGS.lr)
    train_config.train.model_update_conf(_BERT_MODEL_UPDATE_CONF)
    train_config.enable_inplace(enable_inplace)

    @flow.global_function(train_config)
    def PretrainJob():
        loss = BuildPreTrainNet(
            batch_size=FLAGS.batch_size,
            data_part_num=FLAGS.data_part_num,
            seq_length=FLAGS.seq_length,
            max_position_embeddings=FLAGS.max_position_embeddings,
            num_hidden_layers=1,
            num_attention_heads=FLAGS.num_attention_heads,
            hidden_dropout_prob=FLAGS.hidden_dropout_prob,
            attention_probs_dropout_prob=FLAGS.attention_probs_dropout_prob,
            vocab_size=FLAGS.vocab_size,
            type_vocab_size=FLAGS.type_vocab_size,
            max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        )
        flow.losses.add_loss(loss)
        return loss

    check_point = flow.train.CheckPoint()
    check_point.load(FLAGS.model_load_dir)
    ret = [PretrainJob().get().mean() for _ in range(num_iters)]
    flow.clear_default_session()
    return np.array(ret)
