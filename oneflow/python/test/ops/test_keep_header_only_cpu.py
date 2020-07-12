import numpy as np
import oneflow as flow

func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)
func_config.default_distribute_strategy(flow.distribute.consistent_strategy())


def test_keep_header_only_cpu(test_case):
    @flow.global_function(func_config)
    def job(x=flow.FixedTensorDef((2, 3, 4), dtype=flow.float)):
        with flow.fixed_placement("cpu", "0:0"):
            x = flow.identity(x)
            return flow.math.reduced_shape_elem_cnt(x)

    test_case.assertTrue(job(np.zeros((2, 3, 4), np.float32)).get().item() == 2 * 3 * 4)
