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

import uuid
from typing import Callable, Optional, Union

import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.session_context as session_ctx
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.local_blob as local_blob_util
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.watcher as watcher_util
import oneflow.python.framework.typing as oft
import oneflow.python.framework.typing_util as oft_util
import oneflow.python.lib.core.enable_if as enable_if
import oneflow.python.framework.hob as hob
from oneflow.core.job.lbi_diff_watcher_info_pb2 import LbiAndDiffWatcherUuidPair
from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.eager as eager_util
import oneflow
from oneflow_api import ConsistentBlob, MirroredBlob
import oneflow_api
import inspect


@oneflow_export("watch")
def Watch(
    blob_watched: oneflow_api.BlobDesc,
    handler_or_prompt: Optional[Union[Callable, str]] = None,
) -> None:
    r"""Register callback for a blob. The callback function will be called after the computation produce the blob finishes. We can use it to watch the values of Blob.

    Args:
        blob_watched: a `Blob`
        handler_or_prompt: a function has an argument of a `Blob`

    For example:

    Example 1:

    .. code-block:: python

        import oneflow as flow
        import oneflow.typing as tp


        def watch_handler(y: tp.Numpy):
            print("out", y)


        @flow.global_function()
        def watch_Job() -> None:
            init = flow.constant_initializer(2.5)
            variable = flow.get_variable(
                "variable-weight",
                shape=(5, ),
                initializer=init,
                trainable=True
            )
            flow.watch(variable, watch_handler)


        checkpoint = flow.train.CheckPoint()
        checkpoint.init()
        watch_Job()

        # out [2.5 2.5 2.5 2.5 2.5]

    Example 2:

    .. code-block:: python

        import oneflow as flow
        import oneflow.typing as tp
        import numpy as np

        def watch_handler(y: tp.Numpy):
            print("out", y)


        @flow.global_function()
        def watch_Job(x: tp.Numpy.Placeholder((1, 3, 2, 2))
        ) -> None:
            initializer = flow.truncated_normal(0.1)
            conv2d = flow.layers.conv2d(
                x,
                filters=3,
                kernel_size=1,
                strides=1,
                padding='SAME',
                kernel_initializer=initializer,
                name="Conv2d"
            )

            flow.watch(conv2d, watch_handler)


        checkpoint = flow.train.CheckPoint()
        checkpoint.init()
        x = np.ones(shape=(1, 3, 2, 2)).astype(np.float32)
        watch_Job(x)

        # out [[[[ 0.03757111  0.03757111]
        #        [ 0.03757111  0.03757111]]

        #       [[-0.36131713 -0.36131713]
        #        [-0.36131713 -0.36131713]]

        #       [[-0.12266113 -0.12266113]
        #        [-0.12266113 -0.12266113]]]]

    """
    api = enable_if.unique([EagerWatch, LazyWatch])
    return api(blob_watched, handler_or_prompt)


@enable_if.condition(hob.in_global_mode & hob.eager_execution_enabled)
def EagerWatch(blob_watched, handler_or_prompt=None):
    handler = _CheckOrMakeHandler(blob_watched, handler_or_prompt)
    local_blob = local_blob_util.MakeLocalBlob4EagerBlob(blob_watched)
    handler(oft_util.TransformWatchedBlob(local_blob, handler))


@enable_if.condition(hob.in_global_mode & ~hob.eager_execution_enabled)
def LazyWatch(blob_watched, handler_or_prompt=None):
    handler = _CheckOrMakeHandler(blob_watched, handler_or_prompt)
    if isinstance(blob_watched, ConsistentBlob):
        LazyConsistentWatch(blob_watched, handler)
    elif isinstance(blob_watched, MirroredBlob):
        handlers = _MakeSubConsistentBlobHandlers(blob_watched, handler)
        for consistent_blob, sub_handler in zip(
            blob_watched.sub_consistent_blob_list, handlers
        ):
            assert isinstance(consistent_blob, ConsistentBlob)
            LazyConsistentWatch(consistent_blob, sub_handler)
    else:
        raise NotImplementedError


def LazyConsistentWatch(blob_watched, handler):
    handler_uuid = str(uuid.uuid1())
    op_conf = op_conf_util.OperatorConf()
    op_conf.name = id_util.UniqueStr("ForeignWatch_")
    setattr(op_conf.foreign_watch_conf, "in", blob_watched.unique_name)
    op_conf.foreign_watch_conf.handler_uuid = handler_uuid
    device_name = blob_watched.parallel_conf.device_name(0)
    with oneflow.scope.placement("cpu", "0:0"):
        compile_context.CurJobAddOp(op_conf)
    watcher_util.BindUuidAndHandler(handler_uuid, blob_watched, handler)


@oneflow_export("watch_diff")
def WatchDiff(
    blob_watched: oneflow_api.BlobDesc,
    handler_or_prompt: Optional[Union[Callable, str]] = None,
) -> None:
    r"""Register callback for gradient of a blob. The callback will be called after the computation produce the gradient blob finishes.

    Args:
        blob_watched: a `Blob`
        handler_or_prompt: a function has an argument of a `Blob`

    For example:

    Example 1:

    .. code-block:: python

        import oneflow as flow
        import oneflow.typing as tp


        BATCH_SIZE = 20

        def watch_diff_handler(blob: tp.Numpy):
            print("watch_diff_handler:", blob, blob.shape, blob.dtype)

        @flow.global_function(type="train")
        def train_job(
            images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
            labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
        ) -> tp.Numpy:
            initializer = flow.truncated_normal(0.1)
            with flow.scope.placement("gpu", "0:0"):
                reshape = flow.reshape(images, [images.shape[0], -1])
                hidden = flow.layers.dense(
                    reshape,
                    512,
                    activation=flow.nn.relu,
                    kernel_initializer=initializer,
                    name="hidden",
                )
                logits = flow.layers.dense(
                    hidden, 10, kernel_initializer=initializer, name="output"
                )
                loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")

            lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.1])
            flow.optimizer.SGD(lr_scheduler, momentum=0).minimize(loss)
            flow.watch_diff(logits, watch_diff_handler)
            return loss


        if __name__ == "__main__":
            checkpoint = flow.train.CheckPoint()
            checkpoint.init()
            (train_images, train_labels), (test_images, test_labels) = flow.data.load_mnist(
                    BATCH_SIZE
            )
            for i, (images, labels) in enumerate(zip(train_images, train_labels)):
                loss = train_job(images, labels)


        # watch_diff_handler: [[-1.88834548e-01  2.71021971e-03  2.28271242e-02  7.17673637e-03
        #                       4.10183379e-03  8.93106461e-02  2.23669074e-02  3.86103359e-03
        #                       3.12465224e-02  5.23346756e-03] .....

    Example 2:

    .. code-block:: python

        import oneflow as flow
        import oneflow.typing as tp
        import numpy as np


        BATCH_SIZE = 20

        def watch_diff_handler(blob: tp.Numpy):
            print("watch_diff_handler:", blob)


        @flow.global_function(type="train")
        def watch_matmul_diff_job(
            images: tp.Numpy.Placeholder((3, 3), dtype=flow.float),
        ) -> None:
            with flow.scope.placement("cpu", "0:0"):
                weight_initializer = flow.constant_initializer(2)
                weight_shape = (3, BATCH_SIZE)
                weight = flow.get_variable(
                    "matmultest-weight",
                    shape=weight_shape,
                    initializer=weight_initializer)
                output = flow.linalg.matmul(images, weight)

            lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.1])
            flow.optimizer.SGD(lr_scheduler, momentum=0.9).minimize(output)
            flow.watch_diff(weight, watch_diff_handler)


        if __name__ == "__main__":
            check_point = flow.train.CheckPoint()
            check_point.init()

            x = np.array([[1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1]]).astype(np.float32)
            watch_matmul_diff_job(x)

        # watch_diff_handler: [[3. 3. 3.]
        #                      [3. 3. 3.]
        #                      [3. 3. 3.]]

    Example 3:

    .. code-block:: python

        import oneflow as flow
        import oneflow.typing as tp
        import numpy as np


        def watch_diff_handler(blob: tp.Numpy):
            print("watch_diff_handler:", blob, blob.shape, blob.dtype)


        @flow.global_function(type="train")
        def watch_conv_diff_job(
            images: tp.Numpy.Placeholder((1, 1, 4, 4), dtype=flow.float),
        ) -> None:
            with flow.scope.placement("gpu", "0:0"):
                weight_shape = (1, 1, 3, 3)
                weight_initializer = flow.truncated_normal(0.1)
                weight = flow.get_variable(
                    name="conv-weight",
                    shape=weight_shape,
                    initializer=weight_initializer
                )
                output = flow.nn.conv2d(images, weight, strides=1, padding="VALID")

            lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.1])
            flow.optimizer.SGD(lr_scheduler, momentum=0.9).minimize(output)
            flow.watch_diff(weight, watch_diff_handler)


        if __name__ == "__main__":
            check_point = flow.train.CheckPoint()
            check_point.init()

            x = np.array([[[[ 1.,  2.,  3.,  4.],
                            [ 5.,  6.,  7.,  8.],
                            [ 9., 10., 11., 12.],
                            [13., 14., 15., 16.]]]]).astype(np.float32)

            watch_conv_diff_job(x)

        # watch_diff_handler: [[[[14. 18. 22.]
        #                        [30. 34. 38.]
        #                        [46. 50. 54.]]]]

    """
    api = enable_if.unique([EagerWatchDiff, LazyWatchDiff])
    return api(blob_watched, handler_or_prompt)


@enable_if.condition(hob.in_global_mode & hob.eager_execution_enabled)
def EagerWatchDiff(blob_watched, handler_or_prompt=None):
    handler = _CheckOrMakeHandler(blob_watched, handler_or_prompt)
    handler_uuid = str(uuid.uuid1())
    lbi_and_uuid = LbiAndDiffWatcherUuidPair()
    # Copy cfg LBI to proto LBI
    lbi_and_uuid.lbi.op_name = blob_watched.lbi.op_name()
    lbi_and_uuid.lbi.blob_name = blob_watched.lbi.blob_name()
    lbi_and_uuid.watcher_uuid = handler_uuid
    c_api_util.CurJobBuildAndInferCtx_AddLbiAndDiffWatcherUuidPair(lbi_and_uuid)
    uuid2watch_handler = session_ctx.GetDefaultSession().uuid2watch_handler
    uuid2watch_handler[handler_uuid] = lambda x: EagerWatch(x, handler_or_prompt)


@enable_if.condition(hob.in_global_mode & ~hob.eager_execution_enabled)
def LazyWatchDiff(blob_watched, handler_or_prompt=None):
    handler = _CheckOrMakeHandler(blob_watched, handler_or_prompt)
    if isinstance(blob_watched, ConsistentBlob):
        LazyConsistentWatchDiff(blob_watched, handler)
    elif isinstance(blob_watched, MirroredBlob):
        handlers = _MakeSubConsistentBlobHandlers(blob_watched, handler)
        for consistent_blob, sub_handler in zip(
            blob_watched.sub_consistent_blob_list, handlers
        ):
            assert isinstance(consistent_blob, ConsistentBlob)
            LazyConsistentWatchDiff(consistent_blob, sub_handler)
    else:
        raise NotImplementedError


def LazyConsistentWatchDiff(blob_watched, handler):
    handler_uuid = str(uuid.uuid1())
    lbi_and_uuid = LbiAndDiffWatcherUuidPair()
    # Copy cfg LBI to proto LBI
    lbi_and_uuid.lbi.op_name = blob_watched.lbi.op_name()
    lbi_and_uuid.lbi.blob_name = blob_watched.lbi.blob_name()
    lbi_and_uuid.watcher_uuid = handler_uuid
    c_api_util.CurJobBuildAndInferCtx_AddLbiAndDiffWatcherUuidPair(lbi_and_uuid)
    watcher_util.BindUuidAndHandler(handler_uuid, blob_watched, handler)


def _CheckOrMakeHandler(blob_watched, handler_or_prompt):
    if callable(handler_or_prompt):
        parameters = inspect.signature(handler_or_prompt).parameters
        oft_util.CheckWatchCallbackParameterAnnotation(parameters)
        annotation = parameters[list(parameters.keys())[0]].annotation
        oft_util.CheckWatchedBlobByAnnotation(blob_watched, annotation)
        return handler_or_prompt
    prompt = handler_or_prompt

    def Handler(x: GetTypeAnnotation(blob_watched)):
        if prompt is not None:
            print(str(prompt))
        print(x)

    return Handler


def _MakeSubConsistentBlobHandlers(blob_watched, handler):
    assert isinstance(blob_watched, MirroredBlob)
    handler4parallel_id_and_local_blob = _MakeHandler4ParallelIdAndLocalBlob(
        blob_watched, handler
    )
    return [
        _WrapperHandler4ParallelIdAndLocalBlob(i, handler4parallel_id_and_local_blob)
        for i in range(len(blob_watched.sub_consistent_blob_list))
    ]


def _WrapperHandler4ParallelIdAndLocalBlob(
    parallel_id, handler4parallel_id_and_local_blob
):
    return lambda local_blob: handler4parallel_id_and_local_blob(
        parallel_id, local_blob
    )


def _MakeHandler4ParallelIdAndLocalBlob(blob_watched, handler):
    parallel_id2consistent_local_blob = {}
    len_sub_remote_blobs = len(blob_watched.sub_consistent_blob_list)

    def HandlerParallelIdAndLocalBlob(parallel_id, local_blob):
        assert parallel_id not in parallel_id2consistent_local_blob
        parallel_id2consistent_local_blob[parallel_id] = local_blob
        if len(parallel_id2consistent_local_blob) != len_sub_remote_blobs:
            return
        local_blob_list = [
            parallel_id2consistent_local_blob[parallel_id]
            for i in range(len_sub_remote_blobs)
        ]
        assert len(local_blob_list) == 1
        local_blob = local_blob_util.LocalBlob(
            local_blob_list[0].numpy(), blob_watched.is_dynamic
        )
        handler(oft_util.TransformWatchedBlob(local_blob, handler))

    return HandlerParallelIdAndLocalBlob


def GetTypeAnnotation(blob_watched):
    # TODO(chengcheng): oft.Numpy support dynamic
    if not blob_watched.is_dynamic:
        return oft.Numpy
    else:
        return oft.ListNumpy
