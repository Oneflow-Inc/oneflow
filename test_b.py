import oneflow as flow

flow.enable_eager_execution()

x = flow.tensor([0])
y = flow.tensor([1])

z = x + y

print(z)
