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

from alexnet import load_data, alexnet

DEFAULT_BATCH_SIZE = 8
DEFAULT_TRAIN_DATA_PATH = "/dataset/imagenet_227/train/32/"
DEFAULT_TRAIN_DATA_PART_NUM = 32
DEFAULT_INFER_DATA_PATH = "/dataset/imagenet_227/train/32/"
DEFAULT_INFER_DATA_PART_NUM = 32
DEFAULT_CHECKPOINT_DIR = "/dataset/PNGS/cnns_model_for_test/alexnet/models/of_model_bk"


def init_env():
    flow.env.init()
    flow.config.machine_num(1)
    flow.config.cpu_device_num(1)
    flow.config.gpu_device_num(1)
    flow.config.enable_debug_mode(True)


def make_alexnet_train_func(batch_size, data_dir, data_part_num):
    # func_config = flow.FunctionConfig()
    # func_config.default_data_type(flow.float)
    # func_config.cudnn_conv_force_fwd_algo(0)
    # func_config.cudnn_conv_force_bwd_data_algo(1)
    # func_config.cudnn_conv_force_bwd_filter_algo(1)

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
    # print("image_shape:", image_shape)
    # print("label_shape:", label_shape)

    @flow.global_function(type="predict")
    def alexnet_inference(
        image: flow.typing.Numpy.Placeholder(image_shape, dtype=flow.float32),
        label: flow.typing.Numpy.Placeholder(label_shape, dtype=flow.int32),
    ) -> flow.typing.Numpy:
        input_lbns["image"] = image.logical_blob_name
        input_lbns["label"] = label.logical_blob_name
        image = flow.transpose(image, perm=(0, 3, 1, 2))
        # label = flow.identity(label)
        loss = alexnet(image, label, trainable=False)
        output = flow.math.reduce_mean(loss)
        output_lbns["output"] = output.logical_blob_name
        return output

    return alexnet_inference, input_lbns, output_lbns


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
                ).astype(np.float32)
                images.append(image)
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
    def test_alexnet(test_case, batch_size=8, num_batchs=6):
        ofrecord_data_path = os.path.join(DEFAULT_INFER_DATA_PATH, "part-0")
        image_list, label_list = load_data_from_ofrecord(
            num_batchs, batch_size, ofrecord_data_path
        )

        init_env()
        # alexnet_train = make_alexnet_infer_func(
        #     DEFAULT_BATCH_SIZE, DEFAULT_TRAIN_DATA_PATH, DEFAULT_TRAIN_DATA_PART_NUM
        # )
        assert image_list[0].shape[0] == batch_size
        image_size = tuple(image_list[0].shape[1:])
        alexnet_infer, input_lbns, output_lbns = make_alexnet_infer_func(
            batch_size, image_size
        )
        # origin alexnet inference model
        checkpoint = flow.train.CheckPoint()
        checkpoint.load(DEFAULT_CHECKPOINT_DIR)
        print("alexnet inference result:")
        origin_outputs = []
        for i, (image, label) in enumerate(zip(image_list, label_list)):
            output = alexnet_infer(image, label)
            origin_outputs.append(output.item())
            print("iter#{:<6} output:".format(i), output.item())
        origin_outputs = np.array(origin_outputs, dtype=np.float32)

        # save model
        saved_model_path = "alexnet_models"
        if os.path.exists(saved_model_path) and os.path.isdir(saved_model_path):
            shutil.rmtree(saved_model_path)

        model_version = 1
        saved_model_builder = flow.SavedModelBuilderV2(saved_model_path)
        job_builder = (
            saved_model_builder.ModelName("alexnet")
            .Version(model_version)
            .Job(alexnet_infer)
        )
        for input_name, lbn in input_lbns.items():
            job_builder.Input(input_name, lbn)
        for output_name, lbn in output_lbns.items():
            job_builder.Output(output_name, lbn)
        job_builder.Complete().Save()

        # load model and run
        flow.clear_default_session()
        model_meta_file_path = os.path.join(
            "alexnet_models", str(model_version), "saved_model.prototxt"
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

        print("load saved alexnet and inference result:")
        cmp_outputs = []
        for i, (image, label) in enumerate(zip(image_list, label_list)):
            # print("image shape: {}, dtype: {}".format(image.shape, image.dtype))
            # print(
            #     "label shape: {}, dtype: {}, data: {}".format(
            #         label.shape, label.dtype, label
            #     )
            # )
            # if i > 1:
            #     print((image - image_list[i - 1]).mean())
            outputs = sess.run(alexnet_infer.__name__, image=image, label=label)
            cmp_outputs.append(outputs[0].item())
            print("iter#{:<6} output:".format(i), outputs[0].item())
        cmp_outputs = np.array(cmp_outputs, dtype=np.float32)

        test_case.assertTrue(np.allclose(origin_outputs, cmp_outputs))


if __name__ == "__main__":
    unittest.main()
