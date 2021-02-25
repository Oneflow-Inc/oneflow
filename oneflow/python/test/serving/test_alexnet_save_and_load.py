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
import unittest
import shutil
import numpy as np
import google.protobuf.text_format as text_format

import oneflow as flow
import oneflow.core.serving.saved_model_pb2 as saved_model_pb

from alexnet import load_data, alexnet
from ofrecord_dataset import ImageNetRecordDataset

DEFAULT_BATCH_SIZE = 8
DEFAULT_TRAIN_DATA_PATH = "/dataset/imagenet_227/train/32/"
DEFAULT_TRAIN_DATA_PART_NUM = 32
DEFAULT_INFER_DATA_PATH = "/dataset/imagenet_227/train/32/"
DEFAULT_INFER_DATA_PART_NUM = 32
DEFAULT_CHECKPOINT_DIR = "/dataset/PNGS/cnns_model_for_test/alexnet/models/of_model_bk"
DEFAULT_IMAGE_SIZE = 227


def init_env():
    flow.env.init()
    flow.config.machine_num(1)
    flow.config.cpu_device_num(1)
    flow.config.gpu_device_num(1)
    flow.config.enable_debug_mode(True)


def make_alexnet_train_func(batch_size, data_dir, data_part_num):
    @flow.global_function(type="train")
    def alexnet_train() -> flow.typing.Numpy:
        image, label = load_data(batch_size, data_dir, data_part_num)
        loss = alexnet(image, label)
        flow.optimizer.SGD(
            flow.optimizer.PiecewiseConstantScheduler([], [0.00001]), momentum=0
        ).minimize(loss)
        return loss

    return alexnet_train


def make_alexnet_infer_func(batch_size, image_size):
    input_lbns = {}
    output_lbns = {}
    image_shape = (batch_size,) + tuple(image_size)
    label_shape = (batch_size,)

    @flow.global_function(type="predict")
    def alexnet_inference(
        image: flow.typing.Numpy.Placeholder(image_shape, dtype=flow.float32),
        label: flow.typing.Numpy.Placeholder(label_shape, dtype=flow.int32),
    ) -> flow.typing.Numpy:
        input_lbns["image"] = image.logical_blob_name
        input_lbns["label"] = label.logical_blob_name
        image = flow.transpose(image, perm=(0, 3, 1, 2))
        loss = alexnet(image, label, trainable=False)
        # reduce_mean calculate reduce_count in python api, we should only set attribute for op in python,
        # so reduce_count is out of date when we have loaded model and set new batch_size.
        # We will modify implementation of reduce_mean
        # output = flow.math.reduce_mean(loss)
        output = loss
        output_lbns["output"] = output.logical_blob_name
        return output

    return alexnet_inference, input_lbns, output_lbns


def load_saved_model(model_meta_file_path):
    saved_model_proto = saved_model_pb.SavedModel()
    with open(model_meta_file_path, "rb") as f:
        text_format.Merge(f.read(), saved_model_proto)
    return saved_model_proto


@flow.unittest.skip_unless_1n1d()
class TestSaveAndLoadModel(flow.unittest.TestCase):
    def test_alexnet(test_case, batch_size=DEFAULT_BATCH_SIZE, num_batchs=6):
        init_env()
        alexnet_infer, input_lbns, output_lbns = make_alexnet_infer_func(
            batch_size, (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 3)
        )
        flow.load_variables(flow.checkpoint.get(DEFAULT_CHECKPOINT_DIR))

        # save model
        saved_model_path = "alexnet_models"
        model_name = "alexnet"
        model_version = 1

        model_version_path = os.path.join(saved_model_path, str(model_version))
        if os.path.exists(saved_model_path) and os.path.isdir(saved_model_path):
            print(
                "WARNING: The model version path '{}' already exist"
                ", old version directory will be removed".format(model_version_path)
            )
            shutil.rmtree(saved_model_path)

        saved_model_builder = flow.saved_model.ModelBuilder(saved_model_path)
        signature_builder = (
            saved_model_builder.ModelName(model_name)
            .Version(model_version)
            .AddFunction(alexnet_infer)
            .AddSignature("regress")
        )
        for input_name, lbn in input_lbns.items():
            signature_builder.Input(input_name, lbn)
        for output_name, lbn in output_lbns.items():
            signature_builder.Output(output_name, lbn)
        saved_model_builder.Save()

        # test data
        new_batch_size = int(batch_size / 2)
        dataset = ImageNetRecordDataset(
            batch_size=new_batch_size,
            image_resize_size=DEFAULT_IMAGE_SIZE,
            data_format="NHWC",
        )
        image_list, label_list = dataset.load_batchs(num_batchs)
        assert image_list[0].shape[0] == new_batch_size
        image_size = tuple(image_list[0].shape[1:])

        flow.clear_default_session()
        alexnet_infer, _, _ = make_alexnet_infer_func(new_batch_size, image_size)
        flow.load_variables(flow.checkpoint.get(DEFAULT_CHECKPOINT_DIR))
        print("alexnet inference result:")
        origin_outputs = []
        for i, (image, label) in enumerate(zip(image_list, label_list)):
            output = alexnet_infer(image, label)
            # origin_outputs.append(output.item())
            # print("iter#{:<6} output:".format(i), output.item())
            origin_outputs.append(output)
            print("iter#{:<6} output:".format(i), output)

        origin_outputs = np.array(origin_outputs, dtype=np.float32)

        # load model and run
        flow.clear_default_session()
        model_meta_file_path = os.path.join(
            saved_model_path, str(model_version), "saved_model.prototxt"
        )
        saved_model_proto = load_saved_model(model_meta_file_path)
        sess = flow.serving.InferenceSession()
        checkpoint_path = os.path.join(
            saved_model_path, str(model_version), saved_model_proto.checkpoint_dir
        )
        sess.set_checkpoint_path(checkpoint_path)

        graph_name = saved_model_proto.default_graph_name
        graph_def = saved_model_proto.graphs[graph_name]
        signature_def = graph_def.signatures[graph_def.default_signature_name]

        with sess.open(graph_name, signature_def, new_batch_size):
            sess.compile(graph_def.op_list)

        # sess.print_job_set()
        sess.launch()

        job_name = sess.list_jobs()[0]
        input_names = sess.list_inputs()
        print("input names:", input_names)
        for input_name in input_names:
            print(
                'input "{}" info: {}'.format(
                    input_name, sess.input_info(input_name, job_name)
                )
            )
        output_names = sess.list_outputs()
        print("output names:", output_names)
        for output_name in output_names:
            print(
                'output "{}" info: {}'.format(
                    output_name, sess.output_info(output_name, job_name)
                )
            )

        print("load saved alexnet and inference result:")
        print_input_info = False
        cmp_outputs = []
        for i, (image, label) in enumerate(zip(image_list, label_list)):
            if print_input_info:
                print("image shape: {}, dtype: {}".format(image.shape, image.dtype))
                print(
                    "label shape: {}, dtype: {}, data: {}".format(
                        label.shape, label.dtype, label
                    )
                )
                if i > 1:
                    print((image - image_list[i - 1]).mean())

            outputs = sess.run(alexnet_infer.__name__, image=image, label=label)
            # cmp_outputs.append(outputs[0].item())
            # print("iter#{:<6} output:".format(i), outputs[0].item())
            cmp_outputs.append(outputs[0])
            print("iter#{:<6} output:".format(i), outputs[0])

        cmp_outputs = np.array(cmp_outputs, dtype=np.float32)
        test_case.assertTrue(np.allclose(origin_outputs, cmp_outputs))
        sess.close()


if __name__ == "__main__":
    unittest.main()
