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
import numpy as np
import shutil
import unittest
import google.protobuf.text_format as text_format
import oneflow as flow
import oneflow.core.serving.saved_model_pb2 as saved_model_pb

from resnet_model import resnet50
from ofrecord_dataset import ImageNetRecordDataset

DEFAULT_BATCH_SIZE = 4
DEFAULT_CHECKPOINT_DIR = "/dataset/model_zoo/resnet_v15_of_best_model_val_top1_77318"
DEFAULT_IMAGE_SIZE = 224


def init_env():
    flow.env.init()
    flow.config.machine_num(1)
    flow.config.cpu_device_num(1)
    flow.config.gpu_device_num(1)
    flow.config.enable_debug_mode(True)


def make_resnet_infer_func(batch_size, image_size):
    input_lbns = {}
    output_lbns = {}
    image_shape = (batch_size,) + tuple(image_size)

    @flow.global_function(type="predict")
    def resnet_inference(
        image: flow.typing.Numpy.Placeholder(image_shape, dtype=flow.float32)
    ) -> flow.typing.Numpy:
        input_lbns["image"] = image.logical_blob_name
        output = resnet50(image, trainable=False)
        output = flow.nn.softmax(output)
        output_lbns["output"] = output.logical_blob_name
        return output

    return resnet_inference, input_lbns, output_lbns


def load_saved_model(model_meta_file_path):
    saved_model_proto = saved_model_pb.SavedModel()
    with open(model_meta_file_path, "rb") as f:
        text_format.Merge(f.read(), saved_model_proto)
    return saved_model_proto


@flow.unittest.skip_unless_1n1d()
class TestSaveAndLoadModel(flow.unittest.TestCase):
    def test_resnet(test_case, batch_size=DEFAULT_BATCH_SIZE, num_batchs=6):
        init_env()
        # input image format NCHW
        image_size = (3, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
        resnet_infer, input_lbns, output_lbns = make_resnet_infer_func(
            batch_size, image_size
        )

        # resnet inference model parameters
        flow.load_variables(flow.checkpoint.get(DEFAULT_CHECKPOINT_DIR))

        # test data
        dataset = ImageNetRecordDataset(
            batch_size=batch_size,
            image_resize_size=DEFAULT_IMAGE_SIZE,
            data_format="NCHW",
        )
        image_list, label_list = dataset.load_batchs(num_batchs)

        print("resnet inference result:")
        origin_outputs = []
        for i, (image, label) in enumerate(zip(image_list, label_list)):
            output = resnet_infer(image)
            arg_max = np.argmax(output, axis=1)
            origin_outputs.append(arg_max)
            print("iter#{:<6} predict: ".format(i), arg_max, "label: ", label)

        origin_outputs = np.array(origin_outputs, dtype=np.float32)

        # save model
        saved_model_path = "resnet50_models"
        model_version = 1

        model_version_path = os.path.join(saved_model_path, str(model_version))
        if os.path.exists(model_version_path) and os.path.isdir(model_version_path):
            print(
                "WARNING: The model version path '{}' already exist"
                ", old version directory will be removed".format(model_version_path)
            )
            shutil.rmtree(model_version_path)

        saved_model_builder = flow.saved_model.ModelBuilder(saved_model_path)
        signature_builder = (
            saved_model_builder.ModelName("resnet50")
            .Version(model_version)
            .AddFunction(resnet_infer)
            .AddSignature("regress")
        )
        for input_name, lbn in input_lbns.items():
            signature_builder.Input(input_name, lbn)
        for output_name, lbn in output_lbns.items():
            signature_builder.Output(output_name, lbn)
        saved_model_builder.Save()

        # load model and run
        flow.clear_default_session()
        sess = flow.serving.InferenceSession()
        sess.load_saved_model(saved_model_path)
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

        print("load saved resnet and inference result:")
        cmp_outputs = []
        for i, (image, label) in enumerate(zip(image_list, label_list)):
            outputs = sess.run(resnet_infer.__name__, image=image)
            arg_max = np.argmax(outputs[0], axis=1)
            cmp_outputs.append(arg_max)
            print("iter#{:<6} output:".format(i), arg_max, "label: ", label)

        cmp_outputs = np.array(cmp_outputs, dtype=np.float32)
        test_case.assertTrue(np.allclose(origin_outputs, cmp_outputs))
        sess.close()


if __name__ == "__main__":
    unittest.main()
