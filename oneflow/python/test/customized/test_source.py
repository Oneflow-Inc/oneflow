import numpy as np
import oneflow as flow

flow.config.gpu_device_num(1)

func_config = flow.FunctionConfig()
func_config.default_distribute_strategy(flow.scope.consistent_view())
func_config.default_data_type(flow.float)


def my_test_source(name):
    with flow.scope.placement("cpu", "0:0"):
        return (
            flow.user_op_builder(name)
            .Op("TestSource")
            .Output("out")
            .Build()
            .InferAndTryRun()
            .RemoteBlobList()[0]
        )


@flow.global_function(func_config)
def TestSourceJob():
    return my_test_source("my_test_source")


print(TestSourceJob().get())
# 0, 1, 2, 3, 4
