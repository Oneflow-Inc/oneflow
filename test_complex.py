import numpy as np
import oneflow as flow

# np_a = np.array([1.0 + 1j, 2.0, 3.0 - 2j], dtype=np.complex64)
np_a = np.array([[1.0 + 1j, 2.0], [1.0, 2.0 - 1j]], dtype=np.complex128)
# a = flow.from_numpy(np_a)
a = flow.tensor(np_a, dtype=flow.cdouble)
# a = flow.tensor([1.0 + 1j, 2.0], dtype=flow.cfloat, device='cpu')
# a = flow.tensor([1.0 + 1j, 2.0], dtype=flow.cfloat, device='cuda:0')

print('a.shape: ', a.shape, ' a.dtype: ', a.dtype)
print('a: ', a)
print('a.numpy(): ', a.numpy())
print('a[1]: ', a[1])

'''
pass: flow.from_numpy(np_a) np_a: np.complex64 or np.complex128

pass: flow.tensor(np_a or list, dtype=cfloat, device='cpu' or 'cuda')

pass: print a, a.numpy(), a[1] when a.shape=[2, 2]

error: flow.tensor(np_a or list, dtype=flow.cdouble, device='cpu' or 'cuda')

error: print a[1] when a.shape=[2]

find bug: JUST(oneflow::Maybe<std::complex<double>, void>) is not std::complex<double>, please figure out the reason

'''
