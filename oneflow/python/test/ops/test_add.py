import oneflow as flow
import numpy as np

func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)

def test_naive(test_case):
  @flow.function(func_config)
  def AddJob(a=flow.FixedTensorDef((5, 2)), b=flow.FixedTensorDef((5, 2))):
      return a + b + b

  x = np.random.rand(5, 2).astype(np.float32)
  y = np.random.rand(5, 2).astype(np.float32)
  z = None
  z = AddJob(x, y).get().ndarray()
  test_case.assertTrue(np.array_equal(z, x + y + y))

def test_broadcast(test_case):
  flow.config.enable_debug_mode(True)
  @flow.function(func_config)
  def AddJob(a=flow.FixedTensorDef((5, 2)), b=flow.FixedTensorDef((1, 2))):
      return a + b

  x = np.random.rand(5, 2).astype(np.float32)
  y = np.random.rand(1, 2).astype(np.float32)
  z = None
  z = AddJob(x, y).get().ndarray()
  test_case.assertTrue(np.array_equal(z, x + y))
