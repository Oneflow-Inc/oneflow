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
import bert as bert_util
import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util


def PreTrain(
    input_ids_blob,
    input_mask_blob,
    token_type_ids_blob,
    masked_lm_positions_blob,
    masked_lm_ids_blob,
    masked_lm_weights_blob,
    next_sentence_label_blob,
    vocab_size,
    seq_length=512,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=512,
    type_vocab_size=16,
    max_predictions_per_seq=20,
    initializer_range=0.02,
):
    backbone = bert_util.BertBackbone(
        input_ids_blob=input_ids_blob,
        input_mask_blob=input_mask_blob,
        token_type_ids_blob=token_type_ids_blob,
        vocab_size=vocab_size,
        seq_length=seq_length,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        hidden_act=hidden_act,
        hidden_dropout_prob=hidden_dropout_prob,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        max_position_embeddings=max_position_embeddings,
        type_vocab_size=type_vocab_size,
        initializer_range=initializer_range,
    )

    (lm_loss, _, _) = _AddMaskedLanguageModelLoss(
        input_blob=backbone.sequence_output(),
        output_weights_blob=backbone.embedding_table(),
        positions_blob=masked_lm_positions_blob,
        label_id_blob=masked_lm_ids_blob,
        label_weight_blob=masked_lm_weights_blob,
        seq_length=seq_length,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        max_predictions_per_seq=max_predictions_per_seq,
        hidden_act=bert_util.GetActivation(hidden_act),
        initializer_range=initializer_range,
    )
    pooled_output = PooledOutput(
        backbone.sequence_output(), hidden_size, initializer_range
    )
    (ns_loss, _, _) = _AddNextSentenceOutput(
        input_blob=pooled_output,
        label_blob=next_sentence_label_blob,
        hidden_size=hidden_size,
        initializer_range=initializer_range,
    )
    with flow.scope.namespace("cls-loss"):
        total_loss = lm_loss + ns_loss
    return total_loss


def PooledOutput(sequence_output, hidden_size, initializer_range):
    with flow.scope.namespace("bert-pooler"):
        first_token_tensor = flow.slice(sequence_output, [None, 0, 0], [None, 1, -1])
        first_token_tensor = flow.reshape(first_token_tensor, [-1, hidden_size])
        pooled_output = bert_util._FullyConnected(
            first_token_tensor,
            input_size=hidden_size,
            units=hidden_size,
            weight_initializer=bert_util.CreateInitializer(initializer_range),
            name="dense",
        )
        pooled_output = flow.math.tanh(pooled_output)
    return pooled_output


def _AddMaskedLanguageModelLoss(
    input_blob,
    output_weights_blob,
    positions_blob,
    label_id_blob,
    label_weight_blob,
    seq_length,
    hidden_size,
    vocab_size,
    max_predictions_per_seq,
    hidden_act,
    initializer_range,
):

    with flow.scope.namespace("other"):
        sum_label_weight_blob = flow.math.reduce_sum(label_weight_blob, axis=[-1])
        ones = sum_label_weight_blob * 0.0 + 1.0
        sum_label_weight_blob = flow.math.reduce_sum(sum_label_weight_blob)
        batch_size = flow.math.reduce_sum(ones)
        sum_label_weight_blob = sum_label_weight_blob / batch_size
    with flow.scope.namespace("cls-predictions"):
        input_blob = _GatherIndexes(input_blob, positions_blob, seq_length, hidden_size)
        with flow.scope.namespace("transform"):
            if callable(hidden_act):
                act_fn = op_conf_util.kNone
            else:
                act_fn = hidden_act
            input_blob = bert_util._FullyConnected(
                input_blob,
                input_size=hidden_size,
                units=hidden_size,
                activation=act_fn,
                weight_initializer=bert_util.CreateInitializer(initializer_range),
                name="dense",
            )
            if callable(hidden_act):
                input_blob = hidden_act(input_blob)
                input_blob = bert_util._LayerNorm(input_blob, hidden_size)
        output_bias = flow.get_variable(
            name="output_bias",
            shape=[vocab_size],
            dtype=input_blob.dtype,
            initializer=flow.constant_initializer(1.0),
        )
        logit_blob = flow.matmul(input_blob, output_weights_blob, transpose_b=True)
        logit_blob = flow.nn.bias_add(logit_blob, output_bias)
        label_id_blob = flow.reshape(label_id_blob, [-1])
        pre_example_loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logit_blob, labels=label_id_blob
        )
        pre_example_loss = flow.reshape(pre_example_loss, [-1, max_predictions_per_seq])
        numerator = pre_example_loss * label_weight_blob
        with flow.scope.namespace("loss"):
            numerator = flow.math.reduce_sum(numerator, axis=[-1])
            denominator = sum_label_weight_blob + 1e-5
            loss = numerator / denominator
        return loss, pre_example_loss, logit_blob


def _GatherIndexes(sequence_blob, positions_blob, seq_length, hidden_size):
    output = flow.gather(
        params=sequence_blob, indices=positions_blob, axis=2, batch_dims=2
    )
    output = flow.reshape(output, [-1, hidden_size])
    return output


def _AddNextSentenceOutput(input_blob, label_blob, hidden_size, initializer_range):
    with flow.scope.namespace("cls-seq_relationship"):
        output_weight_blob = flow.get_variable(
            name="output_weights",
            shape=[2, hidden_size],
            dtype=input_blob.dtype,
            model_name="weight",
            initializer=bert_util.CreateInitializer(initializer_range),
        )
        output_bias_blob = flow.get_variable(
            name="output_bias",
            shape=[2],
            dtype=input_blob.dtype,
            model_name="bias",
            initializer=flow.constant_initializer(0.0),
        )
        logit_blob = flow.matmul(input_blob, output_weight_blob, transpose_b=True)
        logit_blob = flow.nn.bias_add(logit_blob, output_bias_blob)
        pre_example_loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logit_blob, labels=label_blob
        )
        loss = pre_example_loss
        return loss, pre_example_loss, logit_blob
