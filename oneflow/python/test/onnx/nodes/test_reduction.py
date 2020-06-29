import oneflow as flow
from util import convert_to_onnx_and_check

func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)


def generate_reduction_test(flow_op, *args, **kwargs):
    @flow.function(func_config)
    def job(x=flow.FixedTensorDef((3, 5, 4))):
        return flow_op(x, *args, **kwargs)

    convert_to_onnx_and_check(job)


def test_reduce_sum(test_case):
    generate_reduction_test(flow.math.reduce_sum)


def test_reduce_sum_axis_12(test_case):
    generate_reduction_test(flow.math.reduce_sum, axis=[1, 2])


def test_reduce_sum_axis_12_keepdim(test_case):
    generate_reduction_test(flow.math.reduce_sum, axis=[1, 2], keepdims=True)


def test_reduce_prod(test_case):
    generate_reduction_test(flow.math.reduce_prod)


def test_reduce_prod_axis_12(test_case):
    generate_reduction_test(flow.math.reduce_prod, axis=[1, 2])


def test_reduce_prod_axis_12_keepdim(test_case):
    generate_reduction_test(flow.math.reduce_prod, axis=[1, 2], keepdims=True)


def test_reduce_mean(test_case):
    generate_reduction_test(flow.math.reduce_mean)


def test_reduce_mean_axis_12(test_case):
    generate_reduction_test(flow.math.reduce_mean, axis=[1, 2])


def test_reduce_mean_axis_12_keepdim(test_case):
    generate_reduction_test(flow.math.reduce_mean, axis=[1, 2], keepdims=True)


def test_reduce_min(test_case):
    generate_reduction_test(flow.math.reduce_min)


def test_reduce_min_axis_12(test_case):
    generate_reduction_test(flow.math.reduce_min, axis=[1, 2])


def test_reduce_min_axis_12_keepdim(test_case):
    generate_reduction_test(flow.math.reduce_min, axis=[1, 2], keepdims=True)
