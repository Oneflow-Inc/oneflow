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
import oneflow.core.job.saved_model_pb2 as saved_model_pb
import oneflow.core.record.record_pb2 as record_pb
import unittest
import os
import struct
import cv2
import numpy as np
import shutil
import google.protobuf.text_format as text_format

from resnet_model import resnet50

DEFAULT_BATCH_SIZE = 8
DEFAULT_TRAIN_DATA_PATH = "/dataset/imagenet_224/train/32/"
DEFAULT_TRAIN_DATA_PART_NUM = 32
DEFAULT_INFER_DATA_PATH = "/dataset/imagenet_224/train/32/"
DEFAULT_INFER_DATA_PART_NUM = 32
DEFAULT_CHECKPOINT_DIR = "./resnet_v15_of_best_model_val_top1_77318"


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


def preprocess_image(im):
    rgb_mean = [123.68, 116.779, 103.939]
    rgb_std = [58.393, 57.12, 57.375]
    im = cv2.resize(im.astype("uint8"), (224, 224))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im.astype("float32")
    im = (im - rgb_mean) / rgb_std
    im = np.transpose(im, (2, 0, 1))
    return np.ascontiguousarray(im, "float32")


def load_data_from_ofrecord(num_batchs, batch_size, ofrecord_data_file):
    image_list = []
    label_list = []

    with open(ofrecord_data_file, "rb") as reader:
        for i in range(num_batchs):
            images = []
            labels = []
            num_read = batch_size
            while num_read > 0:
                record_head = reader.read(8)
                if record_head is None or len(record_head) != 8:
                    break

                ofrecord = record_pb.OFRecord()
                ofrecord_byte_size = struct.unpack("q", record_head)[0]
                ofrecord.ParseFromString(reader.read(ofrecord_byte_size))

                image_raw_bytes = ofrecord.feature["encoded"].bytes_list.value[0]
                image = cv2.imdecode(
                    np.frombuffer(image_raw_bytes, np.uint8), cv2.IMREAD_COLOR
                )
                images.append(preprocess_image(image))
                label = ofrecord.feature["class/label"].int32_list.value[0]
                labels.append(label)
                num_read -= 1

            if num_read == 0:
                image_list.append(np.stack(images))
                label_list.append(np.array(labels, dtype=np.int32))
            else:
                break

    return image_list, label_list


def load_saved_model(model_meta_file_path):
    saved_model_proto = saved_model_pb.SavedModel()
    with open(model_meta_file_path, "rb") as f:
        text_format.Merge(f.read(), saved_model_proto)
    return saved_model_proto


@flow.unittest.skip_unless_1n1d()
class TestSaveAndLoadModel(flow.unittest.TestCase):
    def test_resnet(test_case, batch_size=8, num_batchs=6):
        ofrecord_data_path = os.path.join(DEFAULT_INFER_DATA_PATH, "part-0")
        image_list, label_list = load_data_from_ofrecord(
            num_batchs, batch_size, ofrecord_data_path
        )

        init_env()
        assert image_list[0].shape[0] == batch_size
        image_size = tuple(image_list[0].shape[1:])
        resnet_infer, input_lbns, output_lbns = make_resnet_infer_func(
            batch_size, image_size
        )
        # origin resnet inference model
        checkpoint = flow.train.CheckPoint()
        checkpoint.load(DEFAULT_CHECKPOINT_DIR)
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
        if os.path.exists(saved_model_path) and os.path.isdir(saved_model_path):
            shutil.rmtree(saved_model_path)

        model_version = 1
        saved_model_builder = flow.SavedModelBuilderV2(saved_model_path)
        job_builder = (
            saved_model_builder.ModelName("resnet50")
            .Version(model_version)
            .Job(resnet_infer)
        )
        for input_name, lbn in input_lbns.items():
            job_builder.Input(input_name, lbn)
        for output_name, lbn in output_lbns.items():
            job_builder.Output(output_name, lbn)
        job_builder.Complete().Save()

        # load model and run
        flow.clear_default_session()
        model_meta_file_path = os.path.join(
            saved_model_path, str(model_version), "saved_model.prototxt"
        )
        saved_model_proto = load_saved_model(model_meta_file_path)
        sess = flow.SimpleSession()
        checkpoint_path = os.path.join(
            saved_model_path, str(model_version), saved_model_proto.checkpoint_dir[0]
        )
        sess.set_checkpoint_path(checkpoint_path)
        for job_name, signature in saved_model_proto.signatures_v2.items():
            sess.setup_job_signature(job_name, signature)

        for job_name, net in saved_model_proto.graphs.items():
            with sess.open(job_name) as sess:
                sess.compile(net.op)

        # sess.print_job_set()
        sess.launch()
        input_names = sess.list_inputs()
        print("input names:", input_names)
        for input_name in input_names:
            print('input "{}" info: {}'.format(input_name, sess.input_info(input_name)))

        print("load saved resnet and inference result:")
        cmp_outputs = []
        for i, (image, label) in enumerate(zip(image_list, label_list)):
            outputs = sess.run(resnet_infer.__name__, image=image)
            arg_max = np.argmax(outputs[0], axis=1)
            cmp_outputs.append(arg_max)
            print("iter#{:<6} output:".format(i), arg_max, "label: ", label)
        cmp_outputs = np.array(cmp_outputs, dtype=np.float32)

        test_case.assertTrue(np.allclose(origin_outputs, cmp_outputs))


if __name__ == "__main__":
    unittest.main()
