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

import oneflow
import oneflow as flow
import oneflow._oneflow_internal
import oneflow.framework.c_api_util as c_api_util
import oneflow.framework.session_context as session_ctx
import oneflow.unittest
from oneflow.framework.multi_client_session import MultiClientSession


@flow.unittest.skip_unless_1n1d()
class TestFeedInputTensor(unittest.TestCase):
    def test_feed_input_tensor(test_case):
        test_case.assertTrue(oneflow.distributed.is_multi_client())
        test_case.assertTrue(oneflow.framework.env_util.HasAllMultiClientEnvVars())
        x = flow.Tensor(1, 1, 10, 10)
        flow.nn.init.uniform_(x, a=-1.0, b=1.0)
        session = session_ctx.GetDefaultSession()
        test_case.assertTrue(isinstance(session, MultiClientSession))
        session.TryInit()
        with oneflow._oneflow_internal.lazy_mode.guard(True):
            oneflow._oneflow_internal.JobBuildAndInferCtx_Open(
                "cc_test_input_op_expr_job"
            )
            job_conf = (
                oneflow._oneflow_internal.oneflow.core.job.job_conf.JobConfigProto()
            )
            job_conf.set_job_name("cc_test_input_op_expr_job")
            job_conf.mutable_predict_conf()
            c_api_util.CurJobBuildAndInferCtx_SetJobConf(job_conf)
            op_name = "cc_Input_0"
            input_conf = (
                oneflow._oneflow_internal.oneflow.core.operator.op_conf.FeedInputOpConf()
            )
            input_conf.set_in_0("EagerTensorInput")
            input_conf.set_out_0("out_0")
            input_op = oneflow._oneflow_internal.one.FeedInputOpExpr(
                op_name, input_conf, ["in_0"], ["out_0"]
            )
            attrs = oneflow._oneflow_internal.MutableCfgAttrMap()
            out_tensor = input_op.apply([x], attrs)[0]
            test_case.assertEqual(out_tensor.shape, (1, 1, 10, 10))
            test_case.assertTrue(out_tensor.is_lazy)
            test_case.assertTrue(out_tensor.is_local)
            oneflow._oneflow_internal.JobBuildAndInferCtx_Close()


if __name__ == "__main__":
    unittest.main()
