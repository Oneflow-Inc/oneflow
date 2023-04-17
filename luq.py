import oneflow as flow
import numpy as np

shape = (2,3,10)
a = np.random.randn(*shape) + 1.0j * np.random.randn(*shape)
a = a.astype(np.complex64)
flow_tensor = flow.from_numpy(a).cuda()

ret = flow.fft.fft(flow_tensor, dim = -1)