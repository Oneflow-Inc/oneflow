import oneflow as flow
x = flow.ones(32,10).to("npu")
y = flow.randn(10).to("npu")
z = flow._C.reduce_sum_like(x,y,[0])
print(z)