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
from __future__ import absolute_import
from typing import Optional, Sequence
from oneflow.python.oneflow_export import oneflow_export

import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.job.initializer_conf_pb2 as initializer_conf_util
import oneflow.core.job.regularizer_conf_pb2 as regularizer_conf_util
import oneflow.python.framework.distribute as distribute_util
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow_api


@oneflow_export("layers.prelu")
def prelu(
    inputs: oneflow_api.BlobDesc,
    alpha_initializer: Optional[initializer_conf_util.InitializerConf] = None,
    alpha_regularizer: Optional[regularizer_conf_util.RegularizerConf] = None,
    shared_axes: Optional[Sequence[int]] = None,
    trainable: bool = True,
    name: str = "PRelu",
    model_distribute: oneflow_api.sbp_descriptor.SbpDescriptor = oneflow_api.sbp_descriptor.broadcast(),
) -> oneflow_api.BlobDesc:
    r"""The Prelu(Parametric Rectified Linear Unit) activation. 
    
    The :math:`\alpha` is a parameter that can be trained in network

    The equation is

    .. math:: 

        out = max(0, x) + \alpha*min(0, x)

    Args:
        inputs (oneflow_api.BlobDesc): The input Blob. 
        alpha_initializer (Optional[initializer_conf_util.InitializerConf], optional): The initializer of alpha. Defaults to None.
        alpha_regularizer (Optional[regularizer_conf_util.RegularizerConf], optional): The regularizer of alpha. Defaults to None.
        shared_axes (Optional[Sequence[int]], optional): The axis along which to share learnable parameters for the prelu activation function. Defaults to None.
        trainable (bool, optional): Whether to train the parameter :math:`\alpha`. Defaults to True.
        name (str, optional): The name for the operation. Defaults to "PRelu".
        model_distribute (oneflow_api.sbp_descriptor.SbpDescriptor, optional): Define the way to ditribute the model. Defaults to oneflow_api.sbp_descriptor.broadcast().

    Returns:
        oneflow_api.BlobDesc: The activated Blob

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import oneflow.typing as tp

        BATCH_SIZE = 100


        def lenet(data, train=False):
            initializer = flow.truncated_normal(0.1)
            conv1 = flow.layers.conv2d(
                data,
                32,
                5,
                padding="SAME",
                name="conv1",
                kernel_initializer=initializer,
            )
            prelu1 = flow.layers.prelu(conv1,
                                    alpha_initializer=initializer,
                                    shared_axes=[2, 3],
                                    name="Prelu1")
            pool1 = flow.nn.max_pool2d(
                prelu1, ksize=2, strides=2, padding="SAME", name="pool1", data_format="NCHW"
            )
            conv2 = flow.layers.conv2d(
                pool1,
                64,
                5,
                padding="SAME",
                name="conv2",
                kernel_initializer=initializer,
            )
            prelu2 = flow.layers.prelu(conv2,
                                    alpha_initializer=initializer,
                                    shared_axes=[2, 3],
                                    name="Prelu2")
            pool2 = flow.nn.max_pool2d(
                prelu2, ksize=2, strides=2, padding="SAME", name="pool2", data_format="NCHW"
            )
            reshape = flow.reshape(pool2, [pool2.shape[0], -1])
            hidden = flow.layers.dense(
                reshape,
                512,
                activation=flow.nn.relu,
                kernel_initializer=initializer,
                name="dense1",
            )
            if train:
                hidden = flow.nn.dropout(hidden, rate=0.5, name="dropout")
            return flow.layers.dense(hidden, 10, kernel_initializer=initializer, name="dense2")


        @flow.global_function(type="train")
        def train_job(
                images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
                labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
        ) -> tp.Numpy:
            with flow.scope.placement("gpu", "0:0"):
                logits = lenet(images, train=True)
                loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
                    labels, logits, name="softmax_loss"
                )

            lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.1])
            flow.optimizer.SGD(lr_scheduler, momentum=0.9).minimize(loss)
            return loss

    """
    alpha_shape = list(inputs.shape[1:])
    if shared_axes is not None:
        for i in shared_axes:
            assert i >= 1 and i < len(inputs.shape)
            alpha_shape[i - 1] = 1

    if alpha_initializer is None:
        alpha_initializer = flow.constant_initializer(0)

    with flow.scope.namespace(name):
        alpha = flow.get_variable(
            name="alpha",
            shape=alpha_shape,
            dtype=inputs.dtype,
            initializer=alpha_initializer,
            regularizer=alpha_regularizer,
            trainable=trainable,
            distribute=model_distribute,
            reuse=False,
        )

    op = (
        flow.user_op_builder(name)
        .Op("prelu")
        .Input("x", [inputs])
        .Input("alpha", [alpha])
        .Output("y")
        .Build()
    )
    return op.InferAndTryRun().SoleOutputBlob()
