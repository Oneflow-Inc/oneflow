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
import oneflow as flow
import oneflow.typing as tp
from oneflow.python.test.onnx.save.util import convert_to_onnx_and_check


def generate_binary_op_test(flow_op, *args, opset=None, **kwargs):
    @flow.global_function()
    def job1():
        x = flow.get_variable(
            name="x1",
            shape=(2, 3, 4),
            dtype=flow.float,
            initializer=flow.random_uniform_initializer(-10, 10),
        )
        y = flow.get_variable(
            name="y1",
            shape=(1, 3, 1),
            dtype=flow.float,
            initializer=flow.random_uniform_initializer(-10, 10),
        )
        return flow_op(x, y, *args, **kwargs)

    convert_to_onnx_and_check(job1, opset=opset)


def generate_unary_op_test(
    flow_op, *args, opset=None, min_val=-10, max_val=10, **kwargs
):
    @flow.global_function()
    def job1():
        x = flow.get_variable(
            name="x1",
            shape=(2, 3, 4),
            dtype=flow.float,
            initializer=flow.random_uniform_initializer(min_val, max_val),
        )
        return flow_op(x, *args, **kwargs)

    convert_to_onnx_and_check(job1, opset=opset)


def test_mul(test_case):
    generate_binary_op_test(flow.math.multiply)


def test_div(test_case):
    generate_binary_op_test(flow.math.divide)


def test_sub(test_case):
    generate_binary_op_test(flow.math.subtract)


def test_add(test_case):
    generate_binary_op_test(flow.math.add)


def test_abs(test_case):
    generate_unary_op_test(flow.math.abs)


def test_ceil(test_case):
    generate_unary_op_test(flow.math.ceil)


def test_acos(test_case):
    generate_unary_op_test(flow.math.acos, min_val=-1, max_val=1)


def test_asin(test_case):
    generate_unary_op_test(flow.math.asin, min_val=-1, max_val=1)


def test_atan(test_case):
    generate_unary_op_test(flow.math.atan, min_val=-1, max_val=1)


def test_acosh(test_case):
    generate_unary_op_test(flow.math.acosh, min_val=1, max_val=100)


def test_asinh(test_case):
    generate_unary_op_test(flow.math.asinh, min_val=-1, max_val=1)


def test_atanh(test_case):
    generate_unary_op_test(flow.math.atanh, min_val=-1, max_val=1)


def test_sin(test_case):
    generate_unary_op_test(flow.math.sin)


def test_cos(test_case):
    generate_unary_op_test(flow.math.cos)


def test_tan(test_case):
    generate_unary_op_test(flow.math.tan)


def test_sinh(test_case):
    generate_unary_op_test(flow.math.sinh)


def test_cosh(test_case):
    generate_unary_op_test(flow.math.cosh)


def test_tanh_v2(test_case):
    generate_unary_op_test(flow.math.tanh_v2)


def test_tanh(test_case):
    generate_unary_op_test(flow.math.tanh)


def test_erf(test_case):
    generate_unary_op_test(flow.math.erf)


def test_log(test_case):
    generate_unary_op_test(flow.math.log, min_val=0, max_val=100)


def test_floor(test_case):
    generate_unary_op_test(flow.math.floor)


def test_reciprocal(test_case):
    generate_unary_op_test(flow.math.reciprocal)


def test_round(test_case):
    generate_unary_op_test(flow.math.round, opset=11)


def test_rsqrt(test_case):
    generate_unary_op_test(flow.math.rsqrt, min_val=0, max_val=100)


def test_sigmoid_v2(test_case):
    generate_unary_op_test(flow.math.sigmoid_v2)


def test_sigmoid(test_case):
    generate_unary_op_test(flow.math.sigmoid)


def test_sign(test_case):
    generate_unary_op_test(flow.math.sign)


def test_softplus(test_case):
    generate_unary_op_test(flow.math.softplus)


def test_sigmoid(test_case):
    generate_unary_op_test(flow.math.sigmoid)


def test_sqrt(test_case):
    generate_unary_op_test(flow.math.sqrt, min_val=0, max_val=100)


def test_sqaure(test_case):
    generate_unary_op_test(flow.math.square)


def test_maximum(test_case):
    generate_binary_op_test(flow.math.maximum)


def test_minimum(test_case):
    generate_binary_op_test(flow.math.minimum)


def test_equal(test_case):
    generate_binary_op_test(flow.math.equal, opset=11)


def test_not_equal(test_case):
    generate_binary_op_test(flow.math.not_equal, opset=11)


def test_less(test_case):
    generate_binary_op_test(flow.math.less)


def test_greater(test_case):
    generate_binary_op_test(flow.math.greater)


def test_less_equal(test_case):
    generate_binary_op_test(flow.math.less_equal)


def test_greater_equal(test_case):
    generate_binary_op_test(flow.math.greater_equal)


def test_squared_difference(test_case):
    generate_binary_op_test(flow.math.squared_difference)


def test_cast(test_case):
    generate_unary_op_test(flow.cast, dtype=flow.int32)


def test_scalar_mul_int(test_cast):
    generate_unary_op_test(flow.math.multiply, 5)


def test_scalar_mul_float(test_cast):
    generate_unary_op_test(flow.math.multiply, 5.1)


def test_scalar_add_int(test_cast):
    generate_unary_op_test(flow.math.add, 5)


def test_scalar_add_float(test_cast):
    generate_unary_op_test(flow.math.add, 5.1)
