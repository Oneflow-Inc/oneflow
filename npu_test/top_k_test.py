import oneflow as flow
import oneflow.nn as nn

a = flow.tensor([1,2,1]).to('npu')
b = flow._C.top_k_npu(a,1,0)
c = flow._C.top_k_npu(a,1,0)
print(b)
print(c)