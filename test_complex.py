import oneflow as flow


a = flow.tensor([1.0 + 2j, 2.0 - 3j, 1j], dtype=flow.cfloat)
a.requires_grad = True
print("a: ", a)

b = flow.real(a)

print("b: ", b)

c = flow.imag(a)

print("c: ", c)

d = flow.conj(a)

print("d: ", d)

loss = flow.sum(b+c)

loss.backward()
