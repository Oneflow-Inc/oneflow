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
import oneflow as flow
import oneflow.typing as oft
import numpy as np
import time

flow.config.gpu_device_num(4)
func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)
# func_config.indexed_slices_optimizer_conf(
#    dict(include_op_names=dict(op_name=["fc7-weight"]))
# )
num_classes = 1000000
emb_size = 128
batch_size = 256
num_sample = 100000
partial_fc = True
indexed_slice_update = True


@flow.global_function(type="train", function_config=func_config)
def PartialFcJob(
    data: oft.Numpy.Placeholder((batch_size, emb_size), dtype=flow.float),
    labels: oft.Numpy.Placeholder((batch_size,), dtype=flow.int32),
):
    with flow.scope.placement("gpu", "0:0-3"):
        fc7_data_distribute = flow.distribute.split(1)
        fc7_weight = flow.get_variable(
            name="fc7-weight",
            shape=(num_classes, emb_size),
            dtype=flow.float,
            initializer=flow.random_normal_initializer(mean=0.0, stddev=0.01),
            trainable=True,
            model_name="weight",
            distribute=flow.distribute.split(0),
        )
        labels = labels.with_distribute(flow.distribute.broadcast())
        if partial_fc:
            (
                maped_label,
                sampled_label,
                sampled_weight,
            ) = flow.distributed_partial_fc_sample(
                weight=fc7_weight, label=labels, num_sample=num_sample,
            )
            maped_label = maped_label.with_distribute(flow.distribute.broadcast())
            labels = maped_label
            fc7_weight = sampled_weight
        data = flow.parallel_cast(data, distribute=flow.distribute.broadcast())
        fc7 = flow.matmul(a=data, b=fc7_weight, transpose_b=True)
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
            labels, fc7.with_distribute(flow.distribute.split(1)), name="softmax_loss"
        )
        flow.optimizer.SGD(
            flow.optimizer.PiecewiseConstantScheduler([], [1e-4]), momentum=0
        ).minimize(loss)
    return loss


# fake labels
labels = np.random.randint(0, num_classes, size=(batch_size,)).astype(np.int32)
data = np.random.rand(batch_size, emb_size).astype(np.float32)

# OneFlow
check_point = flow.train.CheckPoint()
check_point.init()
start_time = time.time()
for i in range(300):
    if i == 100:
        start_time = time.time()
    loss = PartialFcJob(data, labels).get()
time = time.time() - start_time
print("time", time)
print("loss", loss.numpy().mean())
