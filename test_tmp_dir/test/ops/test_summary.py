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
import oneflow as flow
from collections import OrderedDict
import cv2
import time

from test_util import GenArgList


def _read_images_by_cv(image_files):
    images = [
        cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB).astype(np.uint8)
        for image_file in image_files
    ]
    return [cv2.resize(image, (512, 512)) for image in images]


def summary_demo():
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_logical_view(flow.scope.mirrored_view())
    logdir = "/oneflow/log"

    @flow.global_function(function_config=func_config)
    def CreateWriter():
        flow.summary.create_summary_writer(logdir)

    @flow.global_function(function_config=func_config)
    def ScalarJob(
        value: flow.typing.ListNumpy.Placeholder((1,), dtype=flow.float),
        step: flow.typing.ListNumpy.Placeholder((1,), dtype=flow.int64),
        tag: flow.typing.ListNumpy.Placeholder((1000,), dtype=flow.int8),
    ):
        flow.summary.scalar(value, step, tag)

    @flow.global_function(function_config=func_config)
    def HistogramJob(
        value: flow.typing.ListNumpy.Placeholder((200, 200, 200), dtype=flow.float),
        step: flow.typing.ListNumpy.Placeholder((1,), dtype=flow.int64),
        tag: flow.typing.ListNumpy.Placeholder((9,), dtype=flow.int8),
    ):
        flow.summary.histogram(value, step, tag)

    @flow.global_function(function_config=func_config)
    def PbJob(
        value: flow.typing.ListNumpy.Placeholder((1500,), dtype=flow.int8),
        step: flow.typing.ListNumpy.Placeholder((1,), dtype=flow.int64),
    ):
        flow.summary.pb(value, step=step)

    @flow.global_function(function_config=func_config)
    def ImageJob(
        value: flow.typing.ListNumpy.Placeholder(
            shape=(100, 2000, 2000, 4), dtype=flow.uint8
        ),
        step: flow.typing.ListNumpy.Placeholder((1,), dtype=flow.int64),
        tag: flow.typing.ListNumpy.Placeholder((10,), dtype=flow.int8),
    ):
        flow.summary.image(value, step=step, tag=tag)

    @flow.global_function(function_config=func_config)
    def FlushJob():
        flow.summary.flush_summary_writer()

    CreateWriter()
    projecotr = flow.summary.Projector(logdir)
    projecotr.create_embedding_projector()
    projecotr.create_exception_projector()

    hparams = {
        flow.summary.HParam("learning_rate", flow.summary.RealRange(1e-2, 1e-1)): 0.02,
        flow.summary.HParam("dense_layers", flow.summary.IntegerRange(2, 7)): 5,
        flow.summary.HParam(
            "optimizer", flow.summary.ValueSet(["adam", "sgd"])
        ): "adam",
        flow.summary.HParam("accuracy", flow.summary.RealRange(1e-2, 1e-1)): 0.001,
        flow.summary.HParam("magic", flow.summary.ValueSet([False, True])): True,
        flow.summary.Metric("loss", float): 0.02,
        "dropout": 0.6,
    }

    for i in range(200):
        t = ["vgg16", "resnet50", "mask-rcnn", "yolov3"]
        pb = flow.summary.text(t)
        value = np.fromstring(str(pb), dtype=np.int8)
        step = np.array([i], dtype=np.int64)
        PbJob([value], [step])

        pb2 = flow.summary.hparams(hparams)
        value = np.fromstring(str(pb2), dtype=np.int8)
        step = np.array([i], dtype=np.int64)
        PbJob([value], [step])

    for idx in range(10):
        value = np.array([idx], dtype=np.float32)
        step = np.array([idx], dtype=np.int64)
        tag = np.fromstring("scalar", dtype=np.int8)
        ScalarJob([value], [step], [tag])

    value = np.array(
        [
            [[1, 2, 3, 0], [0, 2, 3, 1], [2, 3, 4, 1]],
            [[1, 0, 2, 0], [2, 1, 2, 0], [2, 1, 1, 1]],
        ],
        dtype=np.float64,
    )

    for idx in range(1):
        value = np.random.rand(100, 100, 100).astype(np.float32)
        step = np.array([idx], dtype=np.int64)
        tag = np.fromstring("histogram", dtype=np.int8)
        HistogramJob([value], [step], [tag])

    value_ = np.random.rand(10, 10, 10).astype(np.float32)
    label = (np.random.rand(10) * 10).astype(np.int64)
    x = (np.random.rand(10, 10, 10) * 255).astype(np.uint8)
    sample_name = "sample"
    sample_type = "image"
    step = 1
    tag_exception = "exception_projector"
    tag_embedding = "embedding_projector"
    projecotr.exception_projector(
        value=value,
        tag=tag_exception,
        step=step,
        sample_name=sample_name,
        sample_type=sample_type,
        x=x,
    )
    projecotr.embedding_projector(
        value=value,
        label=label,
        tag=tag_embedding,
        step=step,
        sample_name=sample_name,
        sample_type=sample_type,
        x=x,
    )

    image1_path = "~/oneflow/image1"
    image2_path = "~/oneflow/image2"

    image_files = [
        image1_path,
        image2_path,
    ]

    images = _read_images_by_cv(image_files)
    images = np.array(images, dtype=np.uint8)
    imageRed = np.ones([512, 512, 3]).astype(np.uint8)
    Red = np.array([0, 255, 255], dtype=np.uint8)
    imageNew = np.multiply(imageRed, Red)
    imageNew = np.expand_dims(imageNew, axis=0)
    images = np.concatenate((images, imageNew), axis=0)
    step = np.array([1], dtype=np.int64)
    tag = np.fromstring("image", dtype=np.int8)
    ImageJob([images], [step], [tag])

    graph = flow.summary.Graph(logdir)
    graph.write_structure_graph()
