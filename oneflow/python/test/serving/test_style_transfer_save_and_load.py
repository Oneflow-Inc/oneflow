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
import cv2
import unittest
import sys
import os
import argparse

import oneflow as flow
import style_model


def init_env():
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


def make_style_transfer(image_height, image_width, channels=3):
    @flow.global_function("predict", get_predict_config())
    def style_model_predict(
        image: flow.typing.Numpy.Placeholder(
            (1, channels, image_height, image_width), dtype=flow.float32
        )
    ) -> flow.typing.Numpy:
        style_out = style_model.styleNet(image, trainable=True)
        return style_out

    return style_model_predict


def load_image(image_file):
    im = cv2.imread(image_file)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = np.transpose(im, (2, 0, 1))
    im = np.expand_dims(im, axis=0)
    return np.ascontiguousarray(im, "float32")


def recover_image(im):
    im = np.squeeze(im)
    im = np.transpose(im, (1, 2, 0))
    im = cv2.cvtColor(np.float32(im), cv2.COLOR_RGB2BGR)
    return im.astype(np.uint8)


@flow.unittest.skip_unless_1n1d()
class TestSaveAndLoadModel(flow.unittest.TestCase):
    INPUT_IMAGE_FILE = (
        "/dataset/model_zoo/fast_neural_style/images/content-images/amber.jpg"
    )
    OUTPUT_IMAGE_FILE = None
    CHECKPOINT_DIR = "/dataset/model_zoo/fast_neural_style/sketch_lr_0.001000_cw_10000.000000_sw_10000000000.000000_epoch_0_iter_4400_loss_3008.877197"

    def test_style_model(self):
        init_env()
        input_image = load_image(self.INPUT_IMAGE_FILE)
        image_height, image_width = input_image.shape[2:]
        style_transfer = make_style_transfer(image_height, image_width)
        flow.load_variables(flow.checkpoint.get(self.CHECKPOINT_DIR))

        # save
        saved_model_path = "style_models"
        model_version = 1
        saved_model_version_dir = os.path.join(saved_model_path, str(model_version))
        if not os.path.exists(saved_model_version_dir):
            saved_model_builder = (
                flow.saved_model.ModelBuilder(saved_model_path)
                .ModelName("style_transfer")
                .Version(model_version)
            )
            saved_model_builder.AddFunction(style_transfer).Finish()
            saved_model_builder.Save()

        flow.clear_default_session()

        # load
        sess = flow.serving.InferenceSession()
        sess.load_saved_model(saved_model_path)
        sess.launch()

        job_names = sess.list_jobs()
        print("job names:", job_names)
        input_names = sess.list_inputs()
        print("input names:", input_names)
        for input_name in input_names:
            print(
                'input "{}" info: {}'.format(
                    input_name, sess.input_info(input_name, job_names[0])
                )
            )
        output_names = sess.list_outputs()
        print("output names:", output_names)
        for output_name in output_names:
            print(
                'input "{}" info: {}'.format(
                    output_name, sess.output_info(output_name, job_names[0])
                )
            )

        input_dict = {input_names[0]: input_image}
        outputs = sess.run(style_transfer.__name__, **input_dict)
        if self.OUTPUT_IMAGE_FILE is not None:
            cv2.imwrite(self.OUTPUT_IMAGE_FILE, recover_image(outputs[0]))
            print("write styled output image to", self.OUTPUT_IMAGE_FILE)

        sess.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-image-file")
    parser.add_argument("--output-image-file")
    parser.add_argument("--model-dir")
    args, unknown = parser.parse_known_args()
    if args.input_image_file is not None:
        TestSaveAndLoadModel.INPUT_IMAGE_FILE = args.input_image_file
    if args.output_image_file is not None:
        TestSaveAndLoadModel.OUTPUT_IMAGE_FILE = args.output_image_file
    if args.model_dir is not None:
        TestSaveAndLoadModel.CHECKPOINT_DIR = args.model_dir

    argv = sys.argv[0:1] + unknown
    unittest.main(argv=argv)
