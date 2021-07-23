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
import oneflow.experimental as flow
import oneflow.python.framework.session_context as session_ctx
import oneflow._oneflow_internal
from oneflow.python.framework.multi_client_session import MultiClientSession
import oneflow.python.framework.c_api_util as c_api_util


@flow.unittest.skip_unless_1n1d()
class TestUserOpGraph(unittest.TestCase):
    def test_user_op_graph(test_case):
        test_case.assertTrue(oneflow.distributed.is_multi_client())
        test_case.assertTrue(
            oneflow.python.framework.env_util.HasAllMultiClientEnvVars()
        )

        x0 = flow.Tensor(20, 30)
        weight0 = flow.Tensor(30, 50)
        x1 = flow.Tensor(50, 70)

        # NOTE(chengcheng): this tiny net is:
        #    x0 * weight0 -> out0
        #    relu(out0) -> y0
        #    y0 * x1 -> out1
        #    relu(out1) -> y1

        flow.nn.init.uniform_(x0, a=-1.0, b=1.0)
        flow.nn.init.uniform_(x1, a=-1.0, b=1.0)
        flow.nn.init.uniform_(weight0, a=-1.0, b=1.0)

        session = session_ctx.GetDefaultSession()
        test_case.assertTrue(isinstance(session, MultiClientSession))
        session.TryInit()

        with oneflow._oneflow_internal.lazy_mode.gard(True):

            oneflow._oneflow_internal.JobBuildAndInferCtx_Open(
                "cc_test_user_op_expr_job"
            )
            job_conf = (
                oneflow._oneflow_internal.oneflow.core.job.job_conf.JobConfigProto()
            )
            job_conf.set_job_name("cc_test_user_op_expr_job")
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

            if not x0.is_determined:
                x0.determine()
            x0_tensor_in_c = x0._local_or_consistent_tensor
            if not x1.is_determined:
                x1.determine()
            x1_tensor_in_c = x1._local_or_consistent_tensor
            if not weight0.is_determined:
                weight0.determine()
            weight0_tensor_in_c = weight0._local_or_consistent_tensor

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


if __name__ == "__main__":
    unittest.main()
