import oneflow
x = oneflow.randn(2).to("npu")
y = oneflow.randn(2,2,)
z = oneflow.randn(2,2,2)
p = oneflow.randn(2,2,2,2)
print(x)
