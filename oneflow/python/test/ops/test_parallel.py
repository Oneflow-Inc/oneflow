import oneflow as flow
import numpy as np

def test_1n1c(test_case):
    flow.config.gpu_device_num(1)
    NaiveTest(test_case)

def test_1n2c(test_case):
    flow.config.gpu_device_num(2)
    NaiveTest(test_case)

@flow.unittest.num_nodes_required(2)
def test_2n2c(test_case):
    flow.config.gpu_device_num(1)
    NaiveTest(test_case)

def NaiveTest(test_case):
    shape = (16, 2)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    @flow.function(func_config)
    def AddJob(a=flow.FixedTensorDef(shape), b=flow.FixedTensorDef(shape)):
        return a + b + b
   
    x = np.random.rand(*shape).astype(np.float32)
    y = np.random.rand(*shape).astype(np.float32)
    z = AddJob(x, y).get().ndarray()
    test_case.assertTrue(np.array_equal(z, x + y + y))

