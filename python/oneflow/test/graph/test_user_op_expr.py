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
import unittest

import numpy as np
from google.protobuf import text_format
import os

import oneflow
import oneflow as flow
import oneflow._oneflow_internal
import oneflow._oneflow_internal._C as _C
import oneflow.framework.c_api_util as c_api_util
import oneflow.framework.session_context as session_ctx
import oneflow.unittest
from oneflow.framework.multi_client_session import MultiClientSession


def _get_c_tensor(t):
    if isinstance(t, oneflow._oneflow_internal.Tensor):
        return t
    else:
        raise NotImplementError


def _test_user_op_graph(test_case, is_cuda):
    x0 = flow.tensor(np.random.rand(20, 30), dtype=flow.float32)
    weight0 = flow.tensor(np.random.rand(30, 50), dtype=flow.float32)
    x1 = flow.tensor(np.random.rand(50, 70), dtype=flow.float32)

    if is_cuda:
        x0 = x0.to(device=flow.device("cuda"))
        weight0 = weight0.to(device=flow.device("cuda"))
        x1 = x1.to(device=flow.device("cuda"))

    # NOTE(chengcheng): this tiny net is:
    #    x0 * weight0 -> out0
    #    relu(out0) -> y0
    #    y0 * x1 -> out1
    #    relu(out1) -> y1

    session = session_ctx.GetDefaultSession()
    test_case.assertTrue(isinstance(session, MultiClientSession))
    session.TryInit()

    with oneflow._oneflow_internal.lazy_mode.guard(True):

        oneflow._oneflow_internal.JobBuildAndInferCtx_Open(
            "cc_test_user_op_expr_job_with_cuda" + str(is_cuda)
        )
        job_conf = oneflow.core.job.job_conf_pb2.JobConfigProto()
        job_conf.job_name = "cc_test_user_op_expr_job_with_cuda" + str(is_cuda)
        job_conf.predict_conf.SetInParent()
        c_api_util.CurJobBuildAndInferCtx_SetJobConf(job_conf)

        x0_conf = oneflow.core.operator.op_conf_pb2.FeedInputOpConf()
        x0_conf.in_0 = "in_0"
        x0_conf.out_0 = "out_0"
        x0_conf_str = text_format.MessageToString(x0_conf)
        x0_op = oneflow._oneflow_internal.one.FeedInputOpExpr(
            "cc_Input_0", x0_conf_str, ["in_0"], ["out_0"]
        )

        x1_conf = oneflow.core.operator.op_conf_pb2.FeedInputOpConf()
        x1_conf.in_0 = "in_0"
        x1_conf.out_0 = "out_0"
        x1_conf_str = text_format.MessageToString(x1_conf)
        x1_op = oneflow._oneflow_internal.one.FeedInputOpExpr(
            "cc_Input_1", x1_conf_str, ["in_0"], ["out_0"]
        )

        weight0_conf = oneflow.core.operator.op_conf_pb2.FeedVariableOpConf()
        weight0_conf.in_0 = "in_0"
        weight0_conf.out_0 = "out_0"
        weight0_conf_str = text_format.MessageToString(weight0_conf)
        weight0_op = oneflow._oneflow_internal.one.FeedVariableOpExpr(
            "cc_Variable_0", weight0_conf_str, ["in_0"], ["out_0"]
        )
        output_conf = oneflow.core.operator.op_conf_pb2.FetchOutputOpConf()
        output_conf.in_0 = "in_0"
        output_conf.out_0 = "out_0"
        output_conf_str = text_format.MessageToString(output_conf)
        output_op = oneflow._oneflow_internal.one.FetchOutputOpExpr(
            "cc_Output_0", output_conf_str, ["in_0"], ["out_0"]
        )

        x0_lazy_tensor = _C.dispatch_feed_input(x0_op, x0)
        x1_lazy_tensor = _C.dispatch_feed_input(x1_op, x1)
        weight0_lazy_tensor = _C.dispatch_feed_input(weight0_op, weight0)

        test_case.assertEqual(x0_lazy_tensor.shape, (20, 30))
        test_case.assertTrue(x0_lazy_tensor.is_lazy)

        test_case.assertEqual(weight0_lazy_tensor.shape, (30, 50))
        test_case.assertTrue(weight0_lazy_tensor.is_lazy)
        test_case.assertEqual(x1_lazy_tensor.shape, (50, 70))
        test_case.assertTrue(x1_lazy_tensor.is_lazy)

        out0 = flow._C.matmul(x0_lazy_tensor, weight0_lazy_tensor)
        test_case.assertEqual(out0.shape, (20, 50))
        test_case.assertTrue(out0.is_lazy)

        y0 = flow._C.relu(out0)
        test_case.assertEqual(y0.shape, (20, 50))
        test_case.assertTrue(y0.is_lazy)

        out1 = flow._C.matmul(y0, x1_lazy_tensor)
        test_case.assertEqual(out1.shape, (20, 70))
        test_case.assertTrue(out1.is_lazy)

        y1 = flow._C.relu(out1)
        test_case.assertEqual(y1.shape, (20, 70))
        test_case.assertTrue(y1.is_lazy)

        eager_output = _C.dispatch_fetch_output(output_op, y1)
        test_case.assertEqual(eager_output.shape, (20, 70))
        test_case.assertTrue(not eager_output.is_lazy)

        if is_cuda:
            test_case.assertTrue(x0_lazy_tensor.is_cuda)
            test_case.assertTrue(x1_lazy_tensor.is_cuda)
            test_case.assertTrue(weight0_lazy_tensor.is_cuda)
            test_case.assertTrue(out0.is_cuda)
            test_case.assertTrue(y0.is_cuda)
            test_case.assertTrue(out1.is_cuda)
            test_case.assertTrue(y1.is_cuda)

        oneflow._oneflow_internal.JobBuildAndInferCtx_Close()


@flow.unittest.skip_unless_1n1d()
class TestUserOpGraph(unittest.TestCase):
    def test_user_op_graph_cpu(test_case):
        _test_user_op_graph(test_case, False)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_user_op_graph_gpu(test_case):
        _test_user_op_graph(test_case, True)


if __name__ == "__main__":
    unittest.main()
