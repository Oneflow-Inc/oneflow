import oneflow as flow
import numpy as np

def test_simple(test_case):
  flow.config.gpu_device_num(1)
  data = np.ones((10,), dtype=np.float32)
  def EqOnes(x): test_case.assertTrue(np.allclose(data, x))
  @flow.function()
  def ReluJob(x = flow.FixedTensorDef((10,))):
      y = flow.keras.activations.relu(x)
      flow.watch(x, EqOnes)
      flow.watch(y, EqOnes)
  ReluJob(data)
