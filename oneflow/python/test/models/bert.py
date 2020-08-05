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
import math

import oneflow as flow
import oneflow.core.common.data_type_pb2 as data_type_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util


class BertBackbone(object):
    def __init__(
        self,
        input_ids_blob,
        input_mask_blob,
        token_type_ids_blob,
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
        initializer_range=0.02,
    ):

        with flow.scope.namespace("bert"):
            with flow.scope.namespace("embeddings"):
                (self.embedding_output_, self.embedding_table_) = _EmbeddingLookup(
                    input_ids_blob=input_ids_blob,
                    vocab_size=vocab_size,
                    embedding_size=hidden_size,
                    initializer_range=initializer_range,
                    word_embedding_name="word_embeddings",
                )
                self.embedding_output_ = _EmbeddingPostprocessor(
                    input_blob=self.embedding_output_,
                    seq_length=seq_length,
                    embedding_size=hidden_size,
                    use_token_type=True,
                    token_type_ids_blob=token_type_ids_blob,
                    token_type_vocab_size=type_vocab_size,
                    token_type_embedding_name="token_type_embeddings",
                    use_position_embeddings=True,
                    position_embedding_name="position_embeddings",
                    initializer_range=initializer_range,
                    max_position_embeddings=max_position_embeddings,
                    dropout_prob=hidden_dropout_prob,
                )
            with flow.scope.namespace("encoder"):
                addr_blob = _CreateAttentionMaskFromInputMask(
                    input_mask_blob,
                    from_seq_length=seq_length,
                    to_seq_length=seq_length,
                )
                self.all_encoder_layers_ = _TransformerModel(
                    input_blob=self.embedding_output_,
                    addr_blob=addr_blob,
                    seq_length=seq_length,
                    hidden_size=hidden_size,
                    num_hidden_layers=num_hidden_layers,
                    num_attention_heads=num_attention_heads,
                    intermediate_size=intermediate_size,
                    intermediate_act_fn=GetActivation(hidden_act),
                    hidden_dropout_prob=hidden_dropout_prob,
                    attention_probs_dropout_prob=attention_probs_dropout_prob,
                    initializer_range=initializer_range,
                    do_return_all_layers=False,
                )
            self.sequence_output_ = self.all_encoder_layers_[-1]

    def embedding_output(self):
        return self.embedding_output_

    def all_encoder_layers(self):
        return self.all_encoder_layers_

    def sequence_output(self):
        return self.sequence_output_

    def embedding_table(self):
        return self.embedding_table_


def CreateInitializer(std):
    return flow.truncated_normal(std)


def _Gelu(in_blob):
    return flow.math.gelu(in_blob)


def _TransformerModel(
    input_blob,
    addr_blob,
    seq_length,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    intermediate_act_fn=_Gelu,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    initializer_range=0.02,
    do_return_all_layers=False,
):

    assert hidden_size % num_attention_heads == 0
    attention_head_size = int(hidden_size / num_attention_heads)
    input_width = hidden_size
    prev_output_blob = flow.reshape(input_blob, (-1, input_width))
    all_layer_output_blobs = []
    for layer_idx in range(num_hidden_layers):
        with flow.scope.namespace("layer_%d" % layer_idx):
            layer_input_blob = prev_output_blob
            with flow.scope.namespace("attention"):
                with flow.scope.namespace("self"):
                    attention_output_blob = _AttentionLayer(
                        from_blob=layer_input_blob,
                        to_blob=layer_input_blob,
                        addr_blob=addr_blob,
                        num_attention_heads=num_attention_heads,
                        size_per_head=attention_head_size,
                        attention_probs_dropout_prob=attention_probs_dropout_prob,
                        initializer_range=initializer_range,
                        do_return_2d_tensor=True,
                        from_seq_length=seq_length,
                        to_seq_length=seq_length,
                    )
                with flow.scope.namespace("output"):
                    attention_output_blob = _FullyConnected(
                        attention_output_blob,
                        input_size=num_attention_heads * attention_head_size,
                        units=hidden_size,
                        weight_initializer=CreateInitializer(initializer_range),
                        name="dense",
                    )
                    attention_output_blob = _Dropout(
                        attention_output_blob, hidden_dropout_prob
                    )
                    attention_output_blob = attention_output_blob + layer_input_blob
                    attention_output_blob = _LayerNorm(
                        attention_output_blob, hidden_size
                    )
            with flow.scope.namespace("intermediate"):
                if callable(intermediate_act_fn):
                    act_fn = op_conf_util.kNone
                else:
                    act_fn = intermediate_act_fn
                intermediate_output_blob = _FullyConnected(
                    attention_output_blob,
                    input_size=num_attention_heads * attention_head_size,
                    units=intermediate_size,
                    activation=act_fn,
                    weight_initializer=CreateInitializer(initializer_range),
                    name="dense",
                )
                if callable(intermediate_act_fn):
                    intermediate_output_blob = intermediate_act_fn(
                        intermediate_output_blob
                    )
            with flow.scope.namespace("output"):
                layer_output_blob = _FullyConnected(
                    intermediate_output_blob,
                    input_size=intermediate_size,
                    units=hidden_size,
                    weight_initializer=CreateInitializer(initializer_range),
                    name="dense",
                )
                layer_output_blob = _Dropout(layer_output_blob, hidden_dropout_prob)
                layer_output_blob = layer_output_blob + attention_output_blob
                layer_output_blob = _LayerNorm(layer_output_blob, hidden_size)
                prev_output_blob = layer_output_blob
                all_layer_output_blobs.append(layer_output_blob)

    input_shape = (-1, seq_length, hidden_size)
    if do_return_all_layers:
        final_output_blobs = []
        for layer_output_blob in all_layer_output_blobs:
            final_output_blob = flow.reshape(layer_output_blob, input_shape)
            final_output_blobs.append(final_output_blob)
        return final_output_blobs
    else:
        final_output_blob = flow.reshape(prev_output_blob, input_shape)
        return [final_output_blob]


def _AttentionLayer(
    from_blob,
    to_blob,
    addr_blob,
    num_attention_heads=1,
    size_per_head=512,
    query_act=op_conf_util.kNone,
    key_act=op_conf_util.kNone,
    value_act=op_conf_util.kNone,
    attention_probs_dropout_prob=0.0,
    initializer_range=0.02,
    do_return_2d_tensor=False,
    batch_size=None,
    from_seq_length=None,
    to_seq_length=None,
):
    def TransposeForScores(input_blob, num_attention_heads, seq_length, width):
        output_blob = flow.reshape(
            input_blob, [-1, seq_length, num_attention_heads, width]
        )
        output_blob = flow.transpose(output_blob, perm=[0, 2, 1, 3])
        return output_blob

    from_blob_2d = flow.reshape(from_blob, [-1, num_attention_heads * size_per_head])
    to_blob_2d = flow.reshape(to_blob, [-1, num_attention_heads * size_per_head])

    query_blob = _FullyConnected(
        from_blob_2d,
        input_size=num_attention_heads * size_per_head,
        units=num_attention_heads * size_per_head,
        activation=query_act,
        name="query",
        weight_initializer=CreateInitializer(initializer_range),
    )

    key_blob = _FullyConnected(
        to_blob_2d,
        input_size=num_attention_heads * size_per_head,
        units=num_attention_heads * size_per_head,
        activation=key_act,
        name="key",
        weight_initializer=CreateInitializer(initializer_range),
    )

    value_blob = _FullyConnected(
        to_blob_2d,
        input_size=num_attention_heads * size_per_head,
        units=num_attention_heads * size_per_head,
        activation=value_act,
        name="value",
        weight_initializer=CreateInitializer(initializer_range),
    )

    query_blob = TransposeForScores(
        query_blob, num_attention_heads, from_seq_length, size_per_head
    )
    key_blob = TransposeForScores(
        key_blob, num_attention_heads, to_seq_length, size_per_head
    )

    attention_scores_blob = flow.matmul(query_blob, key_blob, transpose_b=True)
    attention_scores_blob = attention_scores_blob * (
        1.0 / math.sqrt(float(size_per_head))
    )

    attention_scores_blob = attention_scores_blob + addr_blob
    attention_probs_blob = flow.nn.softmax(attention_scores_blob)
    attention_probs_blob = _Dropout(attention_probs_blob, attention_probs_dropout_prob)

    value_blob = flow.reshape(
        value_blob, [-1, to_seq_length, num_attention_heads, size_per_head]
    )
    value_blob = flow.transpose(value_blob, perm=[0, 2, 1, 3])
    context_blob = flow.matmul(attention_probs_blob, value_blob)
    context_blob = flow.transpose(context_blob, perm=[0, 2, 1, 3])

    if do_return_2d_tensor:
        context_blob = flow.reshape(
            context_blob, [-1, num_attention_heads * size_per_head]
        )
    else:
        context_blob = flow.reshape(
            context_blob, [-1, from_seq_length, num_attention_heads * size_per_head]
        )
    return context_blob


def _FullyConnected(
    input_blob, input_size, units, activation=None, name=None, weight_initializer=None
):
    weight_blob = flow.get_variable(
        name=name + "-weight",
        shape=[input_size, units],
        dtype=input_blob.dtype,
        model_name="weight",
        initializer=weight_initializer,
    )
    bias_blob = flow.get_variable(
        name=name + "-bias",
        shape=[units],
        dtype=input_blob.dtype,
        model_name="bias",
        initializer=flow.constant_initializer(0.0),
    )
    output_blob = flow.matmul(input_blob, weight_blob)
    output_blob = flow.nn.bias_add(output_blob, bias_blob)
    return output_blob


def _Dropout(input_blob, dropout_prob):
    if dropout_prob == 0.0:
        return input_blob
    return flow.nn.dropout(input_blob, rate=dropout_prob)


def _LayerNorm(input_blob, hidden_size):
    return flow.layers.layer_norm(
        input_blob, name="LayerNorm", begin_norm_axis=-1, begin_params_axis=-1
    )


def _CreateAttentionMaskFromInputMask(to_mask_blob, from_seq_length, to_seq_length):
    output = flow.cast(to_mask_blob, dtype=flow.float)
    output = flow.reshape(output, [-1, 1, to_seq_length])
    zeros = flow.constant(0.0, dtype=flow.float, shape=[from_seq_length, to_seq_length])
    attention_mask_blob = zeros + output
    attention_mask_blob = flow.reshape(
        attention_mask_blob, [-1, 1, from_seq_length, to_seq_length]
    )
    attention_mask_blob = flow.cast(attention_mask_blob, dtype=flow.float)
    addr_blob = (attention_mask_blob - 1.0) * 10000.0

    return addr_blob


def _EmbeddingPostprocessor(
    input_blob,
    seq_length,
    embedding_size,
    use_token_type=False,
    token_type_ids_blob=None,
    token_type_vocab_size=16,
    token_type_embedding_name="token_type_embeddings",
    use_position_embeddings=True,
    position_embedding_name="position_embeddings",
    initializer_range=0.02,
    max_position_embeddings=512,
    dropout_prob=0.1,
):
    output = input_blob

    if use_token_type:
        assert token_type_ids_blob is not None
        token_type_table = flow.get_variable(
            name=token_type_embedding_name,
            shape=[token_type_vocab_size, embedding_size],
            dtype=input_blob.dtype,
            initializer=CreateInitializer(initializer_range),
        )
        token_type_embeddings = flow.gather(
            params=token_type_table, indices=token_type_ids_blob, axis=0
        )
        output = output + token_type_embeddings

    if use_position_embeddings:
        position_table = flow.get_variable(
            name=position_embedding_name,
            shape=[1, max_position_embeddings, embedding_size],
            dtype=input_blob.dtype,
            initializer=CreateInitializer(initializer_range),
        )
        assert seq_length <= max_position_embeddings
        if seq_length != max_position_embeddings:
            position_table = flow.slice(
                position_table, begin=[None, 0, 0], size=[None, seq_length, -1]
            )
        output = output + position_table

    output = _LayerNorm(output, embedding_size)
    output = _Dropout(output, dropout_prob)

    return output


def _EmbeddingLookup(
    input_ids_blob,
    vocab_size,
    embedding_size=128,
    initializer_range=0.02,
    word_embedding_name="word_embeddings",
):
    embedding_table = flow.get_variable(
        name=word_embedding_name,
        shape=[vocab_size, embedding_size],
        dtype=flow.float,
        initializer=CreateInitializer(initializer_range),
    )
    output = flow.gather(params=embedding_table, indices=input_ids_blob, axis=0)
    return output, embedding_table


def GetActivation(name):
    if name == "linear":
        return None
    elif name == "relu":
        return flow.math.relu
    elif name == "tanh":
        return flow.math.tanh
    elif name == "gelu":
        return flow.math.gelu
    else:
        raise Exception("unsupported activation")
