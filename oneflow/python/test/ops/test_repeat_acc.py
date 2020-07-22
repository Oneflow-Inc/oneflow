import numpy as np
import oneflow as flow
import oneflow.typing as oft

func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)


def test_repeat_acc(test_case):
    @flow.global_function(func_config)
    def RepeatAccJob(a: oft.Numpy.Placeholder((3, 4))):
        return flow.acc(flow.repeat(a, 3), 3)

    x = np.random.rand(3, 4).astype(np.float32)
    y = RepeatAccJob(x).get().numpy()
    test_case.assertTrue(np.array_equal(y, x * 3))
