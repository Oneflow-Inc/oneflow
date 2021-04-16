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
import unittest
import typing
import oneflow as flow
import test_global_storage

from collections import OrderedDict
from test_util import GenArgList, type_name_to_flow_type


def get_func_conf():
    func_conf = flow.FunctionConfig()
    func_conf.default_placement_scope(flow.scope.placement("gpu", "0:0"))
    return func_conf


def get_lr_scheduler():
    return flow.optimizer.PiecewiseConstantScheduler([], [0.001])


def make_self_attn_qk_v_func(batch_size, seq_len, num_heads, head_size, fused, fp16):
    flow.clear_default_session()
    hidden_size = num_heads * 3 * head_size

    @flow.global_function(type="predict", function_config=get_func_conf())
    def self_attn_qk_v_fw_bw(
        h: flow.typing.Numpy.Placeholder(
            shape=(seq_len, batch_size, hidden_size), dtype=flow.float32
        )
    ) -> typing.Tuple[flow.typing.Numpy, flow.typing.Numpy]:
        var = flow.get_variable(
            "var",
            shape=(1,),
            dtype=flow.float32,
            initializer=flow.constant_initializer(1.0, dtype=flow.float32),
            trainable=True,
        )
        h = h + var

        # save grad
        if fused:
            flow.watch_diff(h, test_global_storage.Setter("h_grad_fused"))
        else:
            flow.watch_diff(h, test_global_storage.Setter("h_grad"))

        if fp16:
            h = flow.amp_white_identity(h)

        alpha = 1.0 / np.sqrt(head_size)

        if fused:
            qmk, v = flow.nn.fused_self_attention_query_mul_key_and_value(
                h, head_size=head_size, alpha=alpha
            )
        else:
            # (s, b, H) -> (s, b, n, 3, h) -> (s, b, n, 1, h) -> (s, b, n, h) -> (b, n, s, h)
            h = flow.reshape(h, (seq_len, batch_size, num_heads, 3, head_size))
            q, k, v = (
                flow.transpose(
                    flow.squeeze(
                        flow.slice(
                            h,
                            begin=[None, None, None, i, None],
                            size=[None, None, None, 1, None],
                        ),
                        axis=[-2],
                    ),
                    perm=[1, 2, 0, 3],
                )
                for i in range(3)
            )
            qmk = flow.matmul(q, k, transpose_b=True, alpha=alpha)

        # calc loss for grad
        # h = flow.matmul(qmk, v)
        # loss = flow.math.reduce_sum(h)
        # flow.optimizer.SGD(get_lr_scheduler(), momentum=0).minimize(loss)

        return qmk, v

    return self_attn_qk_v_fw_bw


def gen_random_input(shape):
    return np.random.rand(*shape).astype(np.float32)


def compare_fused_with_no_fusion(
    test_case, batch_size, seq_len, num_heads, head_size, fp16
):
    hidden_size = num_heads * 3 * head_size

    input = gen_random_input((seq_len, batch_size, hidden_size))

    func = make_self_attn_qk_v_func(
        batch_size, seq_len, num_heads, head_size, True, fp16
    )
    qmk, v = func(input)

    func_ = make_self_attn_qk_v_func(
        batch_size, seq_len, num_heads, head_size, False, fp16
    )
    qmk_, v_ = func_(input)

    diff = qmk - qmk_
    print("")
    print("input:", input)
    print("qmk:", qmk)
    print("qmk_:", qmk_)
    print("diff mean:", diff.mean())
    print("diff max:", diff.max())

    test_case.assertTrue(np.allclose(qmk, qmk_))
    test_case.assertTrue(np.allclose(v, v_))

    h_grad = test_global_storage.Get("h_grad_fused")
    h_grad_ = test_global_storage.Get("h_grad")
    test_case.assertTrue(np.allclose(h_grad, h_grad_))


@flow.unittest.skip_unless_1n1d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestFusedSelfAttentionQueryMulKeyAndValue(flow.unittest.TestCase):
    def test_fp32(self):
        if flow.eager_execution_enabled():
            print("\nSkip under erger mode!")
            return

        compare_fused_with_no_fusion(self, 4, 1024, 12, 64, False)

    def test_fp16(self):
        pass


if __name__ == "__main__":
    unittest.main()
