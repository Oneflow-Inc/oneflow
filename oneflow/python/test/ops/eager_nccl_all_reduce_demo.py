import oneflow as flow
import numpy as np
import sys
import oneflow.typing as oft

flow.config.gpu_device_num(4)

func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)
func_config.default_distribute_strategy(flow.scope.consistent_view())

if __name__ == "__main__":

    @flow.global_function(func_config)
    def test_job(x: oft.Numpy.Placeholder((10000,), dtype=flow.float)):
        return flow.eager_nccl_all_reduce(
            x, parallel_conf=""" device_name: "0:gpu:0-3" """,
        )

    for _ in range(10):
        x = np.random.rand(10000).astype(np.float32)
        y = test_job(x).get()
        print(x)
        print(y)
