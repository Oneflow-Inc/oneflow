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
    # def SummaryJob(value=flow.FixedTensorDef((1,), dtype=data_type_util.kDouble), step=flow.FixedTensorDef((1,), dtype=data_type_util.kInt64),
    #     tag=flow.FixedTensorDef((2,), dtype=data_type_util.kChar)):
    #     with flow.device_prior_placement(device_type, "0:0"):
    #         flow.summary.scalar(value, step, tag)
    def SummaryJob():
        with flow.device_prior_placement(device_type, "0:0"):
            flow.summary.create_summary_writer("/home/zjhushengjian/oneflow")
    
    SummaryJob()
    # OneFlow
    # for idx in range(10):
    #     value = np.array([idx],  dtype=np.float64)
    #     step = np.array([idx],  dtype=np.int64)
    #     tag = np.array(['f','1'], dtype=np.chararray)
    #     SummaryJob(value, step, tag)

    


test("cpu")
