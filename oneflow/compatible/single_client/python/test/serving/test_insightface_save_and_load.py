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
import sys
import numpy as np
import shutil
import unittest
import argparse
import oneflow as flow

from insightface_resnet100 import Resnet100
from ofrecord_dataset import FaceEmoreRecordDataset


def init_env():
    flow.env.init()
    flow.config.machine_num(1)
    flow.config.cpu_device_num(1)
    flow.config.gpu_device_num(1)
    flow.config.enable_debug_mode(True)


def get_predict_config(device_type="gpu", device_num=1, default_data_type=flow.float32):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(default_data_type)
    func_config.default_logical_view(flow.scope.consistent_view())
    func_config.default_placement_scope(
        flow.scope.placement(device_type, "0:0-{}".format(device_num - 1))
    )
    return func_config


def make_insightface_resnet100_func(
    batch_size=1, image_height=112, image_width=112, channels=3
):
    shape = (batch_size, channels, image_height, image_width)

    @flow.global_function(type="predict", function_config=get_predict_config())
    def insightface_resnet100_func(
        image: flow.typing.Numpy.Placeholder(shape),
    ) -> flow.typing.Numpy:
        embedding = Resnet100(image, embedding_size=512, fc_type="FC")
        return embedding

    return insightface_resnet100_func


@flow.unittest.skip_unless_1n1d()
class TestSaveAndLoadModel(flow.unittest.TestCase):
    DATA_DIR = "/dataset/insightface/eval_ofrecord/lfw"
    NUM_DATA_PARTS = 1
    MODEL_DIR = "/dataset/model_zoo/insightface/emore_r100_arcface"
    BATCH_SIZE = 1
    IMAGE_SIZE = 112
    NUM_ITER = 6

    def test_insightface(self):
        init_env()
        # test data
        print("Get data from FaceEmoreRecordDataset")
        dataset = FaceEmoreRecordDataset(
            data_dir=self.DATA_DIR,
            num_data_parts=self.NUM_DATA_PARTS,
            batch_size=self.BATCH_SIZE,
            image_width=self.IMAGE_SIZE,
            image_height=self.IMAGE_SIZE,
            data_format="NCHW",
        )
        image_list, issame_list = dataset.load_batchs(self.NUM_ITER)

        # define inference function
        print("Define inference function for insightface")
        infer_fn = make_insightface_resnet100_func(
            self.BATCH_SIZE, self.IMAGE_SIZE, self.IMAGE_SIZE
        )
        print("Load variables for insightface model")
        flow.load_variables(flow.checkpoint.get(self.MODEL_DIR))

        # call inference function to generate compare result
        print("Call inference function directly")
        features = []
        for i, image in enumerate(image_list):
            feature = infer_fn(image)
            features.append(feature)

        # save model
        print("Save model for insightface")
        saved_model_path = "insightface_models"
        model_version = 1

        model_version_path = os.path.join(saved_model_path, str(model_version))
        if os.path.exists(model_version_path) and os.path.isdir(model_version_path):
            print(
                "WARNING: The model version path '{}' already exist"
                ", old version directory will be removed".format(model_version_path)
            )
            shutil.rmtree(model_version_path)

        saved_model_builder = (
            flow.saved_model.ModelBuilder(saved_model_path)
            .ModelName("insightface")
            .Version(model_version)
        )
        saved_model_builder.AddFunction(infer_fn).Finish()
        saved_model_builder.Save()
        flow.clear_default_session()

        # load model and run
        print("InferenceSession load model")
        flow.clear_default_session()
        sess = flow.serving.InferenceSession()
        sess.load_saved_model(saved_model_path)
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

        print("Run model and compare ")
        for i, (image, feature) in enumerate(zip(image_list, features)):
            input_dict = {input_names[0]: image}
            infer_result = sess.run(job_name, **input_dict)
            self.assertTrue(np.allclose(infer_result, feature))

        sess.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir")
    parser.add_argument("--model-dir")
    args, unknown = parser.parse_known_args()
    if args.data_dir is not None:
        TestSaveAndLoadModel.DATA_DIR = args.data_dir
    if args.model_dir is not None:
        TestSaveAndLoadModel.MODEL_DIR = args.model_dir

    argv = sys.argv[0:1] + unknown
    unittest.main(argv=argv)
