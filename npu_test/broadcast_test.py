import oneflow as flow
flow.manual_seed(1)
use_cpu = False
if use_cpu:
    x = flow.randn(3, 1, 1, 6)
    like_tensor = flow.randn(3, 4, 5, 6)
    broadcast_tensor = flow.broadcast_like(x, like_tensor, broadcast_axes=[1, 2])
    print(broadcast_tensor.shape)
else:
    x = flow.ones(3, 1, 1,dtype = flow.float16).to("npu")
    like_tensor = flow.ones(3, 4, 5,dtype=flow.float16).to("npu")
    broadcast_tensor = flow.broadcast_like(x, like_tensor, broadcast_axes=[1, 2])
    #print(broadcast_tensor.shape)