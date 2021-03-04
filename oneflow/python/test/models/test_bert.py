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
import copy
import sys

import numpy as np
import oneflow as flow
from absl import flags
from pretrain import PreTrain
import unittest
import os

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
    ofrecord = flow.data.ofrecord_reader(
        data_dir, batch_size=batch_size, data_part_num=data_part_num, name="decode",
    )
    input_ids = flow.data.ofrecord_raw_decoder(
        ofrecord, "input_ids", shape=(seq_length,), dtype=flow.int32
    )
    next_sentence_labels = flow.data.ofrecord_raw_decoder(
        ofrecord, "next_sentence_labels", shape=(1,), dtype=flow.int32
    )
    input_mask = flow.data.ofrecord_raw_decoder(
        ofrecord, "input_mask", shape=(seq_length,), dtype=flow.int32
    )
    segment_ids = flow.data.ofrecord_raw_decoder(
        ofrecord, "segment_ids", shape=(seq_length,), dtype=flow.int32
    )
    masked_lm_ids = flow.data.ofrecord_raw_decoder(
        ofrecord, "masked_lm_ids", shape=(max_predictions_per_seq,), dtype=flow.int32
    )
    masked_lm_positions = flow.data.ofrecord_raw_decoder(
        ofrecord,
        "masked_lm_positions",
        shape=(max_predictions_per_seq,),
        dtype=flow.int32,
    )
    masked_lm_weights = flow.data.ofrecord_raw_decoder(
        ofrecord,
        "masked_lm_weights",
        shape=(max_predictions_per_seq,),
        dtype=flow.float,
    )

    return (
        input_ids,
        next_sentence_labels,
        input_mask,
        segment_ids,
        masked_lm_ids,
        masked_lm_positions,
        masked_lm_weights,
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
    if data_part_num == 1:
        with flow.scope.placement("cpu", "0:0"):
            decoders = BertDecoder(
                FLAGS.data_dir,
                batch_size,
                data_part_num,
                seq_length,
                max_predictions_per_seq,
            )
    else:
        assert data_part_num > 1
        decoders = BertDecoder(
            FLAGS.data_dir,
            batch_size,
            data_part_num,
            seq_length,
            max_predictions_per_seq,
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


def CreateOptimizer():
    lr_warmup = flow.optimizer.warmup.linear(1000, 0)
    lr_scheduler = flow.optimizer.PolynomialScheduler(
        FLAGS.lr, 100000, 0.0, warmup=lr_warmup
    )
    return flow.optimizer.AdamW(
        lr_scheduler,
        epsilon=1e-6,
        weight_decay=FLAGS.weight_decay_rate,
        weight_decay_excludes=["bias", "LayerNorm", "layer_norm"],
        grad_clipping=flow.optimizer.grad_clipping.by_global_norm(1.0),
    )


def PretrainJob():
    total_loss = BuildPreTrainNet(
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
    opt = CreateOptimizer()
    opt.minimize(total_loss)
    return total_loss


func_config = flow.FunctionConfig()
func_config.default_logical_view(flow.scope.consistent_view())
func_config.enable_auto_mixed_precision(FLAGS.enable_auto_mixed_precision)


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
def test_1n1c(test_case):
    flow.config.enable_debug_mode(True)
    flow.config.gpu_device_num(1)
    pretrain_job = flow.global_function(type="train", function_config=func_config)(
        PretrainJob
    )
    check_point = flow.train.CheckPoint()
    check_point.load(FLAGS.model_load_dir)
    of_loss = [pretrain_job().get().mean() for _ in range(10)]
    print(of_loss)


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
def test_1n4c(test_case):
    flow.config.gpu_device_num(4)
    pretrain_job = flow.global_function(type="train", function_config=func_config)(
        PretrainJob
    )
    check_point = flow.train.CheckPoint()
    check_point.load(FLAGS.model_load_dir)
    of_loss = [pretrain_job().get().mean() for _ in range(10)]
    print(of_loss)


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.num_nodes_required(2)
def test_2n8c(test_case):
    flow.config.gpu_device_num(4)
    pretrain_job = flow.global_function(type="train", function_config=func_config)(
        PretrainJob
    )
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
    train_config.default_logical_view(flow.scope.consistent_view())
    train_config.enable_inplace(enable_inplace)

    @flow.global_function(type="train", function_config=train_config)
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
        CreateOptimizer().minimize(loss)
        return loss

    check_point = flow.train.CheckPoint()
    check_point.load(FLAGS.model_load_dir)
    ret = [PretrainJob().get().mean() for _ in range(num_iters)]
    flow.clear_default_session()
    return np.array(ret)
