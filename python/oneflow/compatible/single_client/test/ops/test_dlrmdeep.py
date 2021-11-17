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
import numpy as np
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client import typing as oft
import argparse
import datetime
import os
import glob
from sklearn.metrics import roc_auc_score
import numpy as np
import time

func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)
batch_size = 16384
dense_dim = 13
embedding_size = 128
sparse_dim = 26
train_data_dir = "/data/wdl_dataset/wdl_ofrecord/train"
train_data_part_num = 256
train_part_name_suffix_length = 5
eval_data_dir = "/data/wdl_dataset/wdl_ofrecord/val"
eval_data_part_num = 256
eval_part_name_suffix_length = 5
max_iter = 10
eval_interval = 100
eval_batchs = 20
vocab_size = 320000

arch_interaction_itself = 0
offset = 1 if arch_interaction_itself else 0
num_fea = sparse_dim + 1
tril_indices = np.array(
    [j + i * num_fea for i in range(num_fea) for j in range(i + offset)]
)
batch_tril_indices = np.random.randint(
    0, 1, size=(batch_size, tril_indices.shape[0])
).astype(np.int32)
batch_tril_indices[:] = tril_indices


def _data_loader_ofrecord(
    data_dir, data_part_num, batch_size, part_name_suffix_length=-1, shuffle=True
):
    assert data_dir
    print("load ofrecord data form", data_dir)
    ofrecord = flow.data.ofrecord_reader(
        data_dir,
        batch_size=batch_size,
        data_part_num=data_part_num,
        part_name_suffix_length=part_name_suffix_length,
        random_shuffle=shuffle,
        shuffle_after_epoch=shuffle,
    )

    def _blob_decoder(bn, shape, dtype=flow.int32):
        return flow.data.OFRecordRawDecoder(ofrecord, bn, shape=shape, dtype=dtype)

    labels = _blob_decoder("labels", (1,))
    dense_fields = _blob_decoder("dense_fields", (dense_dim,), flow.float)
    wide_sparse_fields = _blob_decoder("wide_sparse_fields", (2,))
    deep_sparse_fields = _blob_decoder("deep_sparse_fields", (sparse_dim,))
    return [labels, dense_fields, wide_sparse_fields, deep_sparse_fields]


def embedding_prefetch(indices, embedding_size, name, embedding_name):
    num_unique_indices, unique_indices, reverse_idx = (
        flow.user_op_builder(name)
        .Op("embedding_prefetch")
        .Input("indices", [indices])
        .Output("num_unique_indices")
        .Output("unique_indices")
        .Output("reverse_idx")
        .Attr("name", embedding_name)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()
    )
    return num_unique_indices, unique_indices, reverse_idx


def embedding_lookup(indices, embedding_size, name, embedding_name, optimizer):
    num_unique_indices, unique_indices, reverse_idx = embedding_prefetch(
        indices, embedding_size, name + "_prefetch", embedding_name
    )
    return (
        flow.user_op_builder(name)
        .Op("embedding_lookup")
        .Input("num_unique_indices", [num_unique_indices])
        .Input("unique_indices", [unique_indices])
        .Input("reverse_idx", [reverse_idx])
        .Output("embeddings")
        .Output("unique_values")
        .Attr("name", embedding_name)
        .Attr("optimizer", optimizer)
        .Attr("embedding_size", embedding_size)
        .Attr("dtype", flow.float)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


def embedding_lookup_var(indices, embedding_size, name, embedding_name, optimizer):
    embedding_table = flow.get_variable(
        name="embedding",
        shape=(vocab_size, embedding_size),
        initializer=flow.random_uniform_initializer(minval=-0.05, maxval=0.05),
    )
    embedding = flow.gather(params=embedding_table, indices=indices)
    return embedding


def interaction(fc, sparse_embedding, tril_indices, name=None):
    (batch_size, d) = fc.shape
    concat_list = [flow.reshape(fc, (batch_size, 1, d)), sparse_embedding]
    T = flow.concat(concat_list, axis=1)
    print("T", T.shape)  # (batch_size, 27, 128)
    Z = flow.matmul(T, flow.transpose(T, perm=(0, 2, 1)))
    print("Z", Z.shape)  # (batch_size, 27, 27)
    out = flow.gather(
        flow.reshape(Z, (batch_size, -1)), tril_indices, axis=1, batch_dims=1
    )
    out = flow.concat([fc, out], axis=1)
    print("out", out.shape)  # (batch_size, 128+351)
    return out


hidden_units1 = [512, 256, 128]
hidden_units2 = [8192, 8192, 512, 256, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192]


def _weight_initializer():
    return flow.variance_scaling_initializer(1.0, "fan_avg", "random_normal")


def _bias_initializer(unit):
    stddev = np.sqrt(1.0 / unit)
    return flow.random_normal_initializer(0.0, stddev)


def _model(sparse_indices, dense, batch_tril_indices):
    sparse_embedding1 = embedding_lookup(
        flow.cast(sparse_indices, dtype=flow.int64),
        embedding_size=embedding_size,
        name="EmbeddingLookup1",
        embedding_name="embedding1",
        optimizer="sgd",
    )
    print("sparse_embedding1", sparse_embedding1.shape)
    dense_features = dense
    for idx, units in enumerate(hidden_units1):
        dense_features = flow.layers.dense(
            dense_features,
            units=units,
            kernel_initializer=_weight_initializer(),
            bias_initializer=_bias_initializer(units),
            activation=flow.math.relu,
            use_bias=True,
            name="fc" + str(idx + 1),
        )
        print("dense layer", "fc" + str(idx + 1))
    interaction1 = interaction(dense_features, sparse_embedding1, batch_tril_indices)
    dense_features = interaction1
    for idx, units in enumerate(hidden_units2):
        dense_features = flow.layers.dense(
            dense_features,
            units=units,
            kernel_initializer=_weight_initializer(),
            bias_initializer=_bias_initializer(units),
            activation=flow.math.relu,
            use_bias=True,
            name="fc" + str(idx + 4),
        )
        print("dense layer", "fc" + str(idx + 4))
    dense_features = flow.layers.dense(
        dense_features,
        units=1,
        kernel_initializer=_weight_initializer(),
        bias_initializer=_bias_initializer(units),
        activation=None,
        use_bias=True,
        name="fc_end",
    )
    return dense_features


@flow.global_function(type="train", function_config=func_config)
def DlrmTrain(
    batch_tril_indices: oft.Numpy.Placeholder(
        batch_tril_indices.shape, dtype=flow.int32
    ),
):
    with flow.scope.placement("gpu", "0:0-3"):
        labels, dense, _, sparse_indices = _data_loader_ofrecord(
            data_dir=train_data_dir,
            data_part_num=train_data_part_num,
            batch_size=batch_size,
            part_name_suffix_length=train_part_name_suffix_length,
            shuffle=True,
        )
    with flow.scope.placement("gpu", "0:0"):
        logits = _model(sparse_indices, dense, batch_tril_indices)
        loss = flow.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        lr_warmup = flow.optimizer.warmup.linear(steps=2750, start_multiplier=0.0)
        lr_scheduler = flow.optimizer.DlrmPolynomialScheduler(
            base_lr=24,
            decay_start=49315,
            decay_steps=27772,
            decay_power=2.0,
            end_lr=0.0,
            warmup=lr_warmup,
        )
        flow.optimizer.SGD(lr_scheduler, momentum=0).minimize(loss)
        return loss, logits


@flow.global_function(type="predict", function_config=func_config)
def DlrmEval(
    batch_tril_indices: oft.Numpy.Placeholder(
        batch_tril_indices.shape, dtype=flow.int32
    ),
):
    with flow.scope.placement("gpu", "0:0"):
        labels, dense, _, sparse_indices = _data_loader_ofrecord(
            data_dir=eval_data_dir,
            data_part_num=eval_data_part_num,
            batch_size=batch_size,
            part_name_suffix_length=eval_part_name_suffix_length,
            shuffle=True,
        )
        logits = _model(sparse_indices, dense, batch_tril_indices)
        loss = flow.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        predict = flow.math.sigmoid(logits)
        return loss, predict, labels


# OneFlow
# labels = np.random.randint(0, 1, size=(64, 1)).astype(np.int32)
# dense = np.random.rand(64, dense_dim).astype(np.float32)
# indices = np.random.randint(0, 100, size=(64, sparse_dim)).astype(np.int64)
# indices = np.fromfile("/data/embedding_test/bin/0.bin", dtype=np.int64)[0:64]


def main():
    check_point = flow.train.CheckPoint()
    check_point.init()
    for i in range(max_iter):
        stime=time.time()
        of_out, logits = DlrmTrain(batch_tril_indices).get()
        print("iter time:", i, time.time()-stime)
        #print("sparse_embedding1", sparse_embedding1.numpy())
        #print("sparse_embedding1", np.where(sparse_embedding1.numpy() == 0))
        #np.save("sparse_embedding", sparse_embedding1.numpy())
        if (i + 1) % eval_interval == 0:
            labels = np.array([[0]])
            preds = np.array([[0]])
            cur_time = time.time()
            eval_loss = 0.0
            for j in range(eval_batchs):
                loss, pred, ref = DlrmEval(batch_tril_indices).get()
                label_ = ref.numpy().astype(np.float32)
                labels = np.concatenate((labels, label_), axis=0)
                preds = np.concatenate((preds, pred.numpy()), axis=0)
                eval_loss += loss.mean()
            auc = roc_auc_score(labels[1:], preds[1:])
            print(i + 1, "eval_loss", eval_loss / eval_batchs, "eval_auc", auc)
    # check_point.save("end_model")


if __name__ == "__main__":
    main()
