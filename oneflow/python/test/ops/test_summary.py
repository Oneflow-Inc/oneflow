import os
import numpy as np
import tensorflow as tf
import oneflow as flow
from collections import OrderedDict
import cv2

from test_util import GenArgList


def _read_images_by_cv(image_files):
    images = [
        cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB).astype(np.uint8)
        for image_file in image_files
    ]
    return [cv2.resize(image, (512, 512)) for image in images]


def test(device_type):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    @flow.global_function(func_config)
    def CreateWriter():
        flow.summary.create_summary_writer("/home/zjhushengjian/oneflow")

    @flow.global_function(func_config)
    def ScalarJob(
        value=flow.MirroredTensorDef((1,), dtype=flow.float),
        step=flow.MirroredTensorDef((1,), dtype=flow.int64),
        tag=flow.MirroredTensorDef((1000,), dtype=flow.int8),
    ):
        with flow.device_prior_placement(device_type, "0:0"):
            flow.summary.scalar(value, step, tag)

    @flow.global_function(func_config)
    def HistogramJob(
        value=flow.MirroredTensorDef((200, 200, 200, 200), dtype=flow.float),
        step=flow.MirroredTensorDef((1,), dtype=flow.int64),
        tag=flow.MirroredTensorDef((9,), dtype=flow.int8),
    ):
        with flow.device_prior_placement(device_type, "0:0"):
            flow.summary.histogram(value, step, tag)

    # OneFlow
    @flow.global_function(func_config)
    def PbJob(
        value=flow.MirroredTensorDef((1000,), dtype=flow.int8),
        step=flow.MirroredTensorDef((1,), dtype=flow.int64),
    ):
        with flow.device_prior_placement(device_type, "0:0"):
            flow.summary.pb(value, step=step)

    @flow.global_function(func_config)
    def ImageJob(
        value=flow.MirroredTensorDef(shape=(100, 2000, 2000, 4), dtype=flow.uint8),
        step=flow.MirroredTensorDef((1,), dtype=flow.int64),
        tag=flow.MirroredTensorDef((10,), dtype=flow.int8),
    ):
        with flow.device_prior_placement(device_type, "0:0"):
            flow.summary.image(value, step=step, tag=tag)

    def ExceptionJob(value, step, tag, sample_name=None, sample_type=None, x=None):
        logdir = "/home/zjhushengjian/oneflow"
        flow.exception_projector(
            value=value,
            step=step,
            tag=tag,
            logdir=logdir,
            sample_name=sample_name,
            sample_type=sample_type,
            x=x,
        )

    def EmbeddingJob(
        value, label, step, tag, sample_name=None, sample_type=None, x=None
    ):
        logdir = "/home/zjhushengjian/oneflow"
        flow.embedding_projector(
            value=value,
            label=label,
            step=step,
            tag=tag,
            logdir=logdir,
            sample_name=sample_name,
            sample_type=sample_type,
            x=x,
        )

    CreateWriter()

    # # write text
    # for i in range(10):
    #     t = ["vgg16", "resnet50", "mask-rcnn", "yolov3"]
    #     pb = flow.text(t)
    #     value = np.array(list(str(pb).encode("ascii")), dtype=np.int8)
    #     step = np.array([i], dtype=np.int64)
    #     PbJob([value], [step])

    # write hparams
    hparams = {
        flow.HParam("learning_rate", flow.RealRange(1e-2, 1e-1)): 0.02,
        flow.HParam("dense_layers", flow.IntegerRange(2, 7)): 5,
        flow.HParam("optimizer", flow.ValueSet(["adam", "sgd"])): "adam",
        flow.HParam("accuracy", flow.RealRange(1e-2, 1e-1)): 0.001,
        flow.HParam("magic", flow.ValueSet([False, True])): True,
        flow.Metric("loss", float): 0.02
        "dropout": 0.6,
    }
    pb2 = flow.hparams(hparams)
    value = np.array(list(str(pb2).encode("ascii")), dtype=np.int8)
    step = np.array([0], dtype=np.int64)
    tag = np.array(list("hparams".encode("ascii")), dtype=np.int8)
    PbJob([value], [step])

    # # write scalar
    # for idx in range(10):
    #     value = np.array([idx], dtype=np.float32)
    #     step = np.array([idx], dtype=np.int64)
    #     tag = np.array(list("scalar".encode("ascii")), dtype=np.int8)
    #     ScalarJob([value], [step], [tag])

    # write histogram
    # value = np.array(
    #     [
    #         [[1, 2, 3, 0], [0, 2, 3, 1], [2, 3, 4, 1]],
    #         [[1, 0, 2, 0], [2, 1, 2, 0], [2, 1, 1, 1]],
    #     ],
    #     dtype=np.float64,
    # )

    for idx in range(1):
        value = np.random.rand(100, 100, 100, 100).astype(np.float32)
        step = np.array([idx], dtype=np.int64)
        tag = np.array(list("histogram".encode("ascii")), dtype=np.int8)
        HistogramJob([value], [step], [tag])

    # flow.exception_projector()

    # write image
    # image_files = [
    #     "/home/zjhushengjian/oneflow/image1.png",
    #     "/home/zjhushengjian/oneflow/Lena.png",
    # ]
    # images = _read_images_by_cv(image_files)
    # images = np.array(images, dtype=np.uint8)
    # # image_shapes = [image.shape for image in images]
    # # print(image_shapes)
    # imageRed = np.ones([512, 512, 3]).astype(np.uint8)
    # Red = np.array([0, 255, 255], dtype=np.uint8)
    # imageNew = np.multiply(imageRed, Red)
    # imageNew = np.expand_dims(imageNew, axis=0)
    # images = np.concatenate((images, imageNew), axis=0)
    # # images1 = (np.random.rand(1, 512, 512, 3) * 100).astype(np.uint8)
    # step = np.array([1], dtype=np.int64)
    # tag = np.array(list("image".encode("ascii")), dtype=np.int8)
    # ImageJob([images], [step], [tag])

    # write summary projectors
    value = np.random.rand(10, 10, 10).astype(np.float32)
    label = (np.random.rand(10) * 10).astype(np.int64)
    x = (np.random.rand(10, 10, 10) * 255).astype(np.uint8)
    sample_name = "sample"
    sample_type = "image"
    step = 1
    tag_exception = "exception_projector"
    tag_embedding = "embedding_projector"
    ExceptionJob(
        value=value,
        step=step,
        tag=tag_exception,
        sample_name=sample_name,
        sample_type=sample_type,
        x=x,
    )
    EmbeddingJob(
        value=value,
        label=label,
        step=step,
        tag=tag_embedding,
        sample_name=sample_name,
        sample_type=sample_type,
        x=x,
    )


test("cpu")
