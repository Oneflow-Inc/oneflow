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
import os

import oneflow
import oneflow as flow
import oneflow._oneflow_internal
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
    test_case.assertTrue(oneflow.distributed.is_multi_client())
    test_case.assertTrue(oneflow.framework.env_util.HasAllMultiClientEnvVars())

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

    with oneflow._oneflow_internal.lazy_mode.gard(True):

        oneflow._oneflow_internal.JobBuildAndInferCtx_Open(
            "cc_test_user_op_expr_job_with_cuda" + str(is_cuda)
        )
        job_conf = oneflow._oneflow_internal.oneflow.core.job.job_conf.JobConfigProto()
        job_conf.set_job_name("cc_test_user_op_expr_job_with_cuda" + str(is_cuda))
        job_conf.mutable_predict_conf()
        c_api_util.CurJobBuildAndInferCtx_SetJobConf(job_conf)

        # input_conf.set_in_0("EagerTensorInput")
        # input_conf.set_out_0("out_0")

        x0_conf = (
            oneflow._oneflow_internal.oneflow.core.operator.op_conf.FeedInputOpConf()
        )
        x0_op = oneflow._oneflow_internal.one.FeedInputOpExpr(
            "cc_Input_0", x0_conf, ["in_0"], ["out_0"]
        )
        x1_conf = (
            oneflow._oneflow_internal.oneflow.core.operator.op_conf.FeedInputOpConf()
        )
        x1_op = oneflow._oneflow_internal.one.FeedInputOpExpr(
            "cc_Input_1", x1_conf, ["in_0"], ["out_0"]
        )
        weight0_conf = (
            oneflow._oneflow_internal.oneflow.core.operator.op_conf.FeedVariableOpConf()
        )
        weight0_op = oneflow._oneflow_internal.one.FeedVariableOpExpr(
            "cc_Variable_0", weight0_conf, ["in_0"], ["out_0"]
        )
        output_conf = (
            oneflow._oneflow_internal.oneflow.core.operator.op_conf.FetchOutputOpConf()
        )
        output_op = oneflow._oneflow_internal.one.FetchOutputOpExpr(
            "cc_Output_0", output_conf, ["in_0"], ["out_0"]
        )

        attrs = oneflow._oneflow_internal.MutableCfgAttrMap()

        x0_tensor_in_c = _get_c_tensor(x0)
        x1_tensor_in_c = _get_c_tensor(x1)
        weight0_tensor_in_c = _get_c_tensor(weight0)

        x0_lazy_tensor = x0_op.apply([x0_tensor_in_c], attrs)[0]
        x1_lazy_tensor = x1_op.apply([x1_tensor_in_c], attrs)[0]
        weight0_lazy_tensor = weight0_op.apply([weight0_tensor_in_c], attrs)[0]

        test_case.assertEqual(x0_lazy_tensor.shape, (20, 30))
        test_case.assertTrue(x0_lazy_tensor.is_lazy)
        test_case.assertEqual(weight0_lazy_tensor.shape, (30, 50))
        test_case.assertTrue(weight0_lazy_tensor.is_lazy)
        test_case.assertEqual(x1_lazy_tensor.shape, (50, 70))
        test_case.assertTrue(x1_lazy_tensor.is_lazy)

        out0 = flow.F.matmul(x0_lazy_tensor, weight0_lazy_tensor)
        test_case.assertEqual(out0.shape, (20, 50))
        test_case.assertTrue(out0.is_lazy)

        y0 = flow.F.relu(out0)
        test_case.assertEqual(y0.shape, (20, 50))
        test_case.assertTrue(y0.is_lazy)

        out1 = flow.F.matmul(y0, x1_lazy_tensor)
        test_case.assertEqual(out1.shape, (20, 70))
        test_case.assertTrue(out1.is_lazy)

        y1 = flow.F.relu(out1)
        test_case.assertEqual(y1.shape, (20, 70))
        test_case.assertTrue(y1.is_lazy)

        eager_output = output_op.apply([y1], attrs)[0]
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
