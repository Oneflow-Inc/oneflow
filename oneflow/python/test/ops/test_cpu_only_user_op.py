import oneflow as flow
import numpy as np


def _cpu_only_relu(x):
    op = (
        flow.user_op_builder("CpuOnlyRelu")
        .Op("cpu_only_relu_test")
        .Input("in", [x])
        .Output("out")
        .Build()
    )
    return op.InferAndTryRun().SoleOutputBlob()


def _check_cpu_only_relu_device(test_case, verbose=False):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_placement_scope(flow.device_prior_placement("gpu", "0:0"))

    @flow.global_function(func_config)
    def cpu_only_relu_job(x_def=flow.FixedTensorDef(shape=(2, 5), dtype=flow.float)):
        y = _cpu_only_relu(x_def)
        if verbose:
            print("cpu_only_relu output devices", y.parallel_conf.device_name)
        for device in y.parallel_conf.device_name:
            test_case.assertTrue("cpu" in device)
        return y

    cpu_only_relu_job(np.random.rand(2, 5).astype(np.single)).get()


def _check_non_cpu_only_relu_device(test_case):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_placement_scope(flow.device_prior_placement("gpu", "0:0"))

    @flow.global_function(func_config)
    def relu_job(x_def=flow.FixedTensorDef(shape=(2, 5), dtype=flow.float)):
        with flow.device_prior_placement("gpu", "0:0"):
            y = flow.math.relu(x_def)

        for device in y.parallel_conf.device_name:
            test_case.assertTrue("gpu" in device)

        return y

    relu_job(np.random.rand(2, 5).astype(np.single)).get()


def test_cpu_only_user_op(test_case):
    _check_cpu_only_relu_device(test_case)


def test_non_cpu_only_user_op(test_case):
    _check_non_cpu_only_relu_device(test_case)
