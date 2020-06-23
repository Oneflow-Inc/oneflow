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

    @flow.function(func_config)
    def HistogramJob(value=flow.FixedTensorDef((2,3,4), dtype=data_type_util.kDouble), step=flow.FixedTensorDef((1,), dtype=data_type_util.kInt64),
        tag=flow.FixedTensorDef((9,), dtype=data_type_util.kInt8)):
        with flow.device_prior_placement(device_type, "0:0"):     
            flow.summary.histogram(value, step, tag)

    @flow.function(func_config)
    def TextJob(value=flow.MirroredTensorDef((6,), dtype=data_type_util.kInt8), step=flow.MirroredTensorDef((1,), dtype=data_type_util.kInt64),
        tag=flow.MirroredTensorDef((10,), dtype=data_type_util.kInt8)):
        with flow.device_prior_placement(device_type, "0:0"):     
            flow.summary.text(value, step, tag)
  
    
    # OneFlow
    @flow.function(func_config)
    def PbJob(value=flow.MirroredTensorDef((1000,), dtype=data_type_util.kInt8), step=flow.MirroredTensorDef((1,), dtype=data_type_util.kInt64)):
        with flow.device_prior_placement(device_type, "0:0"):     
            flow.summary.pb(value, step=step)

    CreateWriter()
    # write text
    t = ["laohu", "laoli", "laowang", "laozhang"]
    pb = flow.text(t)
    value = np.array(list(str(pb).encode("ascii")), dtype=np.int8)
    step = np.array([1],  dtype=np.int64)
    PbJob([value], [step])

    # write hparams
    hparams = {
        flow.HParam("learning_rate", flow.RealInterval(1e-2, 1e-1)): 0.02,
        flow.HParam("dense_layers", flow.IntInterval(2, 7)): 5,
        flow.HParam("optimizer", flow.Discrete(["adam", "sgd"])): "adam",
        flow.HParam("accuracy", flow.RealInterval(1e-2, 1e-1)): 0.001,
        flow.HParam(
            "magic",
            flow.Discrete([False, True]),
            display_name="~*~ Magic ~*~",
            description="descriptive",
        ): True,
        "dropout": 0.3,
    }
    pb2 = flow.hparams_pb(hparams)
    value = np.array(list(str(pb2).encode("ascii")), dtype=np.int8)
    step = np.array([1],  dtype=np.int64)
    tag = np.array(list("hparams".encode("ascii")), dtype=np.int8)
    PbJob([value], [step])
     
    # write scalar 
    for idx in range(10):
        value = np.array([idx], dtype=np.float64)
        step = np.array([idx],  dtype=np.int64)
        tag = np.array(list("scalar".encode("ascii")), dtype=np.int8)
        ScalarJob([value], [step], [tag])

    # write histogram
    value = np.array([[[1,2,3,0],[0,2,3,1],[2,3,4,1]],[[1,0,2,0],[2,1,2,0],[2,1,1,1]]],  dtype=np.float64)
    step = np.array([1],  dtype=np.int64)
    tag = np.array(list("histogram".encode("ascii")), dtype=np.int8)
    HistogramJob(value, step, tag)

    
test("cpu")
