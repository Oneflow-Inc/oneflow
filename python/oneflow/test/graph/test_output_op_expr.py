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
import unittest

import numpy as np
from google.protobuf import text_format

import oneflow
import oneflow as flow
import oneflow._oneflow_internal
import oneflow._oneflow_internal._C as _C
import oneflow.framework.c_api_util as c_api_util
import oneflow.framework.session_context as session_ctx
import oneflow.unittest
from oneflow.framework.multi_client_session import MultiClientSession


@flow.unittest.skip_unless_1n1d()
class TestFetchOutputTensor(unittest.TestCase):
    def test_fetch_output_tensor(test_case):
        x = flow.Tensor(1, 1, 10, 10)
        flow.nn.init.uniform_(x, a=-1.0, b=1.0)
        session = session_ctx.GetDefaultSession()
        test_case.assertTrue(isinstance(session, MultiClientSession))
        session.TryInit()
        with oneflow._oneflow_internal.lazy_mode.guard(True):
            oneflow._oneflow_internal.JobBuildAndInferCtx_Open(
                "cc_test_output_op_expr_job"
            )
            job_conf = oneflow.core.job.job_conf_pb2.JobConfigProto()
            job_conf.job_name = "cc_test_output_op_expr_job"
            job_conf.predict_conf.SetInParent()
            c_api_util.CurJobBuildAndInferCtx_SetJobConf(job_conf)
            input_conf = oneflow.core.operator.op_conf_pb2.FeedInputOpConf()
            input_conf.in_0 = "EagerTensorInput"
            input_conf.out_0 = "out_0"
            input_conf_str = text_format.MessageToString(input_conf)
            input_op = oneflow._oneflow_internal.one.FeedInputOpExpr(
                "cc_Input_0", input_conf_str, ["in_0"], ["out_0"]
            )
            output_conf = oneflow.core.operator.op_conf_pb2.FetchOutputOpConf()
            output_conf.in_0 = "LazyTensorInput"
            output_conf.out_0 = "out_0"
            output_conf_str = text_format.MessageToString(output_conf)
            output_op = oneflow._oneflow_internal.one.FetchOutputOpExpr(
                "cc_Output_0", output_conf_str, ["in_0"], ["out_0"]
            )
            lazy_tensor = _C.dispatch_feed_input(input_op, x)
            test_case.assertEqual(lazy_tensor.shape, (1, 1, 10, 10))
            test_case.assertTrue(lazy_tensor.is_lazy)
            test_case.assertTrue(lazy_tensor.is_local)
            eager_tensor = _C.dispatch_fetch_output(output_op, lazy_tensor)
            test_case.assertEqual(eager_tensor.shape, (1, 1, 10, 10))
            test_case.assertTrue(not eager_tensor.is_lazy)
            test_case.assertTrue(eager_tensor.is_local)
            oneflow._oneflow_internal.JobBuildAndInferCtx_Close()


if __name__ == "__main__":
    unittest.main()
