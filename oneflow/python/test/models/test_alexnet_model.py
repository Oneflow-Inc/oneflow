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
import os
from datetime import datetime
import unittest

import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.job.initializer_conf_pb2 as initializer_conf_util


_DATA_DIR = "/dataset/PNGS/PNG227/of_record_repeated"
_MODEL_SAVE_DIR = "./model_save-{}".format(
    str(datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
)
_MODEL_LOAD = "/dataset/PNGS/cnns_model_for_test/alexnet/models/of_model_bk"


class DLNetSpec(object):
    def __init__(self):
        self.batch_size = 8
        self.data_part_num = 32
        self.eval_dir = _DATA_DIR
        self.train_dir = _DATA_DIR
        self.model_save_dir = _MODEL_SAVE_DIR
        self.model_load_dir = _MODEL_LOAD
        self.num_nodes = 1
        self.node_list = None
        self.gpu_num_per_node = 1
        self.iter_num = 10


global_specs = DLNetSpec()


class TrainData(flow.nn.DataModule):
    def __init__(self, specs):
        super().__init__()
        self.specs = specs

    def forward(self):
        return _data_load_layer(self.specs, self.specs.train_dir)


class ValData(flow.nn.DataModule):
    def __init__(self, specs):
        super().__init__()
        self.specs = specs

    def forward(self):
        return _data_load_layer(self.specs, self.specs.eval_dir)


class AlexNet(flow.model.Model):
    def __init__(self, specs, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.specs = specs

    def forward(self, images, trainable=False):
        conv1 = _conv2d_layer(
            "conv1", images, filters=64, kernel_size=11, strides=4, padding="VALID",
        )

        pool1 = flow.nn.avg_pool2d(conv1, 3, 2, "VALID", "NCHW", name="pool1")

        conv2 = _conv2d_layer("conv2", pool1, filters=192, kernel_size=5)

        pool2 = flow.nn.avg_pool2d(conv2, 3, 2, "VALID", "NCHW", name="pool2")

        conv3 = _conv2d_layer("conv3", pool2, filters=384)

        conv4 = _conv2d_layer("conv4", conv3, filters=384)

        conv5 = _conv2d_layer("conv5", conv4, filters=256)

        pool5 = flow.nn.avg_pool2d(conv5, 3, 2, "VALID", "NCHW", name="pool5")

        def _get_initializer():
            kernel_initializer = initializer_conf_util.InitializerConf()
            kernel_initializer.truncated_normal_conf.std = 0.816496580927726
            return kernel_initializer

        if len(pool5.shape) > 2:
            pool5 = flow.reshape(pool5, shape=(pool5.shape[0], -1))

        fc1 = flow.layers.dense(
            inputs=pool5,
            units=4096,
            activation=flow.math.relu,
            use_bias=False,
            kernel_initializer=_get_initializer(),
            bias_initializer=False,
            trainable=trainable,
            name="fc1",
        )

        dropout1 = fc1

        fc2 = flow.layers.dense(
            inputs=dropout1,
            units=4096,
            activation=flow.math.relu,
            use_bias=False,
            kernel_initializer=_get_initializer(),
            bias_initializer=False,
            trainable=trainable,
            name="fc2",
        )

        dropout2 = fc2

        fc3 = flow.layers.dense(
            inputs=dropout2,
            units=1001,
            activation=None,
            use_bias=False,
            kernel_initializer=_get_initializer(),
            bias_initializer=False,
            trainable=trainable,
            name="fc3",
        )

        return fc3

    def training_step(self, batch, optimizer_idx):
        assert optimizer_idx == 0
        images, labels = batch
        fc3 = self(images, True)
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
            labels, fc3, name="softmax_loss"
        )
        return loss

    def validation_step(self, batch):
        images, labels = batch
        fc3 = self(images, False)
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
            labels, fc3, name="softmax_loss"
        )
        return loss

    def configure_optimizers(self):
        return flow.optimizer.SGD(
            flow.optimizer.PiecewiseConstantScheduler([], [0.00001]), momentum=0
        )


class LossMoniter(flow.model.Callback):
    def on_training_step_end(self, step_idx, outputs, optimizer_idx):
        assert optimizer_idx == 0
        loss = outputs.mean()
        fmt_str = "{:>12}  {:>12}  {:>12.6f}"
        print(fmt_str.format(step_idx, "train loss:", loss))

    def on_validation_step_end(self, step_idx, outputs):
        loss = outputs.mean()
        fmt_str = "{:>12}  {:>12}  {:>12.6f}"
        print(fmt_str.format(step_idx, "validation loss:", loss))


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
def test_1n1c(test_case):

    flow.env.ctrl_port(9788)
    flow.config.machine_num(global_specs.num_nodes)
    flow.config.gpu_device_num(global_specs.gpu_num_per_node)

    train_config = flow.ExecutionConfig()
    train_config.default_logical_view(flow.scope.consistent_view())
    train_config.default_data_type(flow.float)

    val_config = flow.ExecutionConfig()
    val_config.default_data_type(flow.float)

    ck_config = flow.model.CheckpointConfig(
        load_dirpath=global_specs.model_load_dir,
        save_dirpath=global_specs.model_save_dir,
        save_interval=10,
    )

    loss_monitor = LossMoniter()

    alexnet_md = AlexNet(
        global_specs,
        is_function_style=True,
        training_config=train_config,
        validation_config=val_config,
        callbacks=[loss_monitor],
    )

    train_data = TrainData(global_specs)
    val_data = ValData(global_specs)

    alexnet_md.fit(
        training_data=train_data,
        validation_data=val_data,
        validation_interval=10,
        checkpoint_config=ck_config,
        max_steps=20,
    )


def _conv2d_layer(
    name,
    input,
    filters,
    kernel_size=3,
    strides=1,
    padding="SAME",
    data_format="NCHW",
    dilation_rate=1,
    activation=op_conf_util.kRelu,
    use_bias=False,
    weight_initializer=flow.random_uniform_initializer(),
    bias_initializer=flow.random_uniform_initializer(),
):
    weight_shape = (filters, input.shape[1], kernel_size, kernel_size)
    weight = flow.get_variable(
        name + "-weight",
        shape=weight_shape,
        dtype=input.dtype,
        initializer=weight_initializer,
    )
    output = flow.nn.conv2d(
        input, weight, strides, padding, data_format, dilation_rate, name=name
    )
    if use_bias:
        bias = flow.get_variable(
            name + "-bias",
            shape=(filters,),
            dtype=input.dtype,
            initializer=bias_initializer,
        )
        output = flow.nn.bias_add(output, bias, data_format)

    if activation is not None:
        if activation == op_conf_util.kRelu:
            output = flow.nn.relu(output)
        else:
            raise NotImplementedError

    return output


def _data_load_layer(args, data_dir):
    node_num = args.num_nodes
    total_batch_size = args.batch_size * args.gpu_num_per_node * node_num
    rgb_mean = [123.68, 116.78, 103.94]
    (image, label) = flow.data.ofrecord_image_classification_reader(
        data_dir,
        batch_size=total_batch_size,
        data_part_num=args.data_part_num,
        image_feature_name="encoded",
        label_feature_name="class/label",
        color_space="RGB",
        name="decode",
    )
    rsz = flow.image.resize(image, target_size=[227, 227], color_space="RGB")
    normal = flow.image.crop_mirror_normalize(
        rsz,
        color_space="RGB",
        output_layout="NCHW",
        mean=rgb_mean,
        output_dtype=flow.float,
    )
    return (normal, label)
