import numpy as np
import oneflow as flow


def test_deadlock(test_case):
    flow.config.gpu_device_num(2)
    func_config = flow.FunctionConfig()
    func_config.enable_inplace(False)

    @flow.global_function(func_config)
    def DistributeConcat():
        with flow.device_prior_placement("gpu", "0:0"):
            w = flow.get_variable(
                "w", (2, 5), initializer=flow.constant_initializer(10)
            )
            x = w + 1
            y = w + 1
        ret = flow.advanced.distribute_concat([x, y])
        # return ret

    DistributeConcat()
