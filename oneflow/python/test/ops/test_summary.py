import os
import numpy as np
import tensorflow as tf
import oneflow as flow
from collections import OrderedDict
import oneflow.core.common.data_type_pb2 as data_type_util


from test_util import GenArgList


def test(device_type):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.double)
    @flow.function(func_config)
    def CreateWriter():
        flow.summary.create_summary_writer("/home/zjhushengjian/oneflow")

    @flow.function(func_config)
    def ScalarJob(value=flow.MirroredTensorDef((1,), dtype=data_type_util.kDouble), step=flow.MirroredTensorDef((1,), dtype=data_type_util.kInt64),
        tag=flow.MirroredTensorDef((20,), dtype=data_type_util.kInt8)):
        with flow.device_prior_placement(device_type, "0:0"):
            flow.summary.scalar(value, step, tag)

    # @flow.function(func_config)
    # def HistogramJob(value=flow.FixedTensorDef((2,3,4), dtype=data_type_util.kDouble), step=flow.FixedTensorDef((1,), dtype=data_type_util.kInt64),
    #     tag=flow.FixedTensorDef((3,), dtype=data_type_util.kInt8)):
    #     with flow.device_prior_placement(device_type, "0:0"):     
    #         flow.summary.histogram(value, step, tag)

    @flow.function(func_config)
    def TextJob(value=flow.MirroredTensorDef((6,), dtype=data_type_util.kInt8), step=flow.MirroredTensorDef((1,), dtype=data_type_util.kInt64),
        tag=flow.MirroredTensorDef((3,), dtype=data_type_util.kInt8)):
        with flow.device_prior_placement(device_type, "0:0"):     
            flow.summary.text(value, step, tag)
    # def SummaryJob():
    #     with flow.device_prior_placement(device_type, "0:0"):
    #         flow.summary.create_summary_writer("/home/zjhushengjian")
    
    # SummaryJob()
    # OneFlow
    @flow.function(func_config)
    def PbJob(value=flow.MirroredTensorDef((1000,), dtype=data_type_util.kInt8), step=flow.MirroredTensorDef((1,), dtype=data_type_util.kInt64)):
        with flow.device_prior_placement(device_type, "0:0"):     
            flow.summary.pb(value, step=step)

    CreateWriter()
    hparams = {
        flow.HParam("learning_rate", flow.RealInterval(1e-2, 1e-1)): 0.02,
        flow.HParam("dense_layers", flow.IntInterval(2, 7)): 5,
        flow.HParam("optimizer", flow.Discrete(["adam", "sgd"])): "adam",
        flow.HParam("who_knows_what"): "???",
        flow.HParam(
            "magic",
            flow.Discrete([False, True]),
            display_name="~*~ Magic ~*~",
            description="descriptive",
        ): True,
        "dropout": 0.3,
    }

    # hparams = {
    #         flow.HParam("learning_rate", flow.RealInterval(1e-2, 1e-1)): 0.02,
    #     }
    pb = flow.hparams_pb(hparams)
    str(pb)

    print(len(str(pb)))
    value = np.array(list(str(pb).encode("ascii")), dtype=np.int8)
    step = np.array([1],  dtype=np.int64)
    PbJob([value], [step])

    # for idx in range(10):
    #     value = np.array(list("laohu".encode("ascii")), dtype=np.int8)
    #     step = np.array([idx],  dtype=np.int64)
    #     tag = np.array(list("hua".encode("ascii")), dtype=np.int8)
    #     ScalarJob([value], [step])

    


test("cpu")
