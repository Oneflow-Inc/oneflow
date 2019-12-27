import oneflow as flow
import numpy as np

func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)


def test_unpack_pack(test_case):
    @flow.function(func_config)
    def UnpackPackJob(a=flow.FixedTensorDef((3, 4))):
        return flow.pack(flow.unpack(a, 3), 3)

    x = np.random.rand(3, 4).astype(np.float32)
    y = UnpackPackJob(x).get().ndarray()
    test_case.assertTrue(np.array_equal(y, x))
