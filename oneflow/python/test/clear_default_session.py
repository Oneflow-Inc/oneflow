import oneflow as flow
import numpy as np

flow.env.data_port(10000)
flow.env.ctrl_port(20000)

def Foo():
  flow.clear_default_session()
  flow.config.gpu_device_num(1)

  func_config = flow.function_config()
  func_config.default_data_type(flow.float)
  @flow.function(func_config)
  def Bar(x = flow.input_blob_def((2, 5))):
      return flow.math.relu(x)

  index = [-2, -1, 0, 1, 2]
  data = []
  for i in index: data.append(np.ones((2, 5), dtype=np.float32) * i)
  for x in data:  print(Bar(x).get())

Foo()
Foo()
