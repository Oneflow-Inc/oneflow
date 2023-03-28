import oneflow as flow


a = flow.tensor([1.0 + 2j, 2.0 - 3j, 1j], dtype=flow.cfloat, device='cuda')
a.requires_grad = True
print("a: ", a)

b = flow.conj(a)

print("b: ", b)

c = a + b

print("c: ", c)

d = c.real()

print("d: ", d)

e = c.imag()

print("e: ", e)

loss = flow.sum(d+e)

loss.backward()
