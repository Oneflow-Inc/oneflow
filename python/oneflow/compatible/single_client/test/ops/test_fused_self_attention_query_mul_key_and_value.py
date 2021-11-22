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
import typing
import unittest

import numpy as np
import test_global_storage

import oneflow.compatible.single_client.unittest
from oneflow.compatible import single_client as flow


def get_func_conf():
    func_conf = flow.FunctionConfig()
    func_conf.default_placement_scope(flow.scope.placement("gpu", "0:0"))
    return func_conf


def get_lr_scheduler():
    return flow.optimizer.PiecewiseConstantScheduler([], [0.001])


def get_alpha(head_size):
    return 1.0


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
        h = h * var
        if fused:
            flow.watch_diff(h, test_global_storage.Setter("h_grad_fused"))
        else:
            flow.watch_diff(h, test_global_storage.Setter("h_grad"))
        if fp16:
            h = flow.amp_white_identity(h)
        alpha = get_alpha(head_size)
        if fused:
            (qmk, v) = flow.nn.fused_self_attention_query_mul_key_and_value(
                h, head_size=head_size, alpha=alpha
            )
        else:
            h = flow.reshape(h, (seq_len, batch_size, -1, 3 * head_size))
            (q, k, v) = (
                flow.transpose(
                    flow.slice(
                        h,
                        begin=[None, None, None, head_size * i],
                        size=[None, None, None, head_size],
                    ),
                    perm=[1, 2, 0, 3],
                )
                for i in range(3)
            )
            qmk = flow.matmul(q, k, transpose_b=True, alpha=alpha)
        h = flow.matmul(qmk, v)
        loss = flow.math.reduce_sum(h)
        flow.optimizer.SGD(get_lr_scheduler(), momentum=0).minimize(loss)
        return (qmk, v)

    return self_attn_qk_v_fw_bw


def gen_random_input(shape):
    return np.random.rand(*shape).astype(np.float32)


def compare_fused_with_no_fused(
    test_case, batch_size, seq_len, num_heads, head_size, fp16, verbose=False
):
    hidden_size = num_heads * 3 * head_size
    input = gen_random_input((seq_len, batch_size, hidden_size))
    func = make_self_attn_qk_v_func(
        batch_size, seq_len, num_heads, head_size, True, fp16
    )
    (qmk, v) = func(input)
    func_ = make_self_attn_qk_v_func(
        batch_size, seq_len, num_heads, head_size, False, fp16
    )
    (qmk_, v_) = func_(input)
    (_q, _k, _v) = np_qkv(input, head_size)
    _qmk = np_bgemm(
        _q.transpose(1, 2, 0, 3), _k.transpose(1, 2, 3, 0), get_alpha(head_size)
    )
    _v = _v.transpose(1, 2, 0, 3)
    if verbose:
        print("")
        print("=" * 80)
        print(f"input: {input.shape}\n{input}")
        print(f"_q: {_q.shape}\n{_q}")
        print(f"_k: {_k.shape}\n{_k}")
        print(f"_v: {_v.shape}\n{_v}")
        print(f"_qmk: {_qmk.shape}\n{_qmk}")
        print(f"qmk: {qmk.shape}\n{qmk}")
        print(f"qmk_: {qmk_.shape}\n{qmk_}")
        diff = qmk - qmk_
        print("abs diff mean:", np.abs(diff).mean())
        print("abs diff max:", np.abs(diff).max())
    test_case.assertTrue(np.allclose(qmk, qmk_))
    test_case.assertTrue(np.allclose(qmk, _qmk))
    test_case.assertTrue(np.allclose(v, v_))
    test_case.assertTrue(np.allclose(v, _v))
    h_grad = test_global_storage.Get("h_grad_fused")
    h_grad_ = test_global_storage.Get("h_grad")
    if verbose:
        print(f"h_grad: {h_grad.shape}\n{h_grad}")
        print(f"h_grad_: {h_grad_.shape}\n{h_grad_}")
    test_case.assertTrue(np.allclose(h_grad, h_grad_))


def np_qkv(h, head_size):
    h = np.reshape(h, (h.shape[0], h.shape[1], -1, 3 * head_size))
    q = h[:, :, :, :head_size]
    k = h[:, :, :, head_size : head_size * 2]
    v = h[:, :, :, head_size * 2 :]
    return (q, k, v)


def np_bgemm(a, b, alpha):
    assert a.ndim == b.ndim
    assert a.ndim >= 2
    assert a.shape[-1] == b.shape[-2]
    if a.ndim > 2:
        a_ = np.reshape(a, (-1, a.shape[-2], a.shape[-1]))
        b_ = np.reshape(b, (-1, b.shape[-2], b.shape[-1]))
        assert a_.shape[0] == b_.shape[0]
        c = np.zeros(shape=(a_.shape[0], a_.shape[-2], b_.shape[-1]), dtype=np.float32)
        for i in range(a_.shape[0]):
            c[i] = np.matmul(a_[i], b_[i]) * alpha
    else:
        c = np.matmul(a, b) * alpha
    shape = a.shape[:-2] + c.shape[-2:]
    return np.reshape(c, shape)


@flow.unittest.skip_unless_1n1d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestFusedSelfAttentionQueryMulKeyAndValue(flow.unittest.TestCase):
    def test_fp32(self):
        if flow.eager_execution_enabled():
            print("\nSkip under erger mode!")
            return
        compare_fused_with_no_fused(self, 4, 1024, 12, 64, False)

    def test_fp16(self):
        if flow.eager_execution_enabled():
            print("\nSkip under erger mode!")
            return
        compare_fused_with_no_fused(self, 4, 1024, 12, 64, True)


if __name__ == "__main__":
    unittest.main()
