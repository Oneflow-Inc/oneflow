"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from oneflow.framework.tensor import Tensor
import oneflow as flow


def fft(input, n=None, dim=-1, norm=None) -> Tensor:
    r"""
    
    Computes the one dimensional discrete Fourier transform of :attr:`input`.
    
    The interface is consistent with PyTorch.
    The documentation is referenced from: https://pytorch.org/docs/2.0/generated/torch.fft.fft2.html.

    Note:
    The Fourier domain representation of any real signal satisfies the
    Hermitian property: `X[i] = conj(X[-i])`. This function always returns both
    the positive and negative frequency terms even though, for real inputs, the
    negative frequencies are redundant. :func:`~torch.fft.rfft` returns the
    more compact one-sided representation where only the positive frequencies
    are returned.

    Args:
        input (Tensor): the input tensor
        n (int, optional): Signal length. If given, the input will either be zero-padded
            or trimmed to this length before computing the FFT.
        dim (int, optional): The dimension along which to take the one dimensional FFT.
        norm (str, optional): Normalization mode. For the forward transform
            (:func:`~torch.fft.fft`), these correspond to:

            * ``"forward"`` - normalize by ``1/n``
            * ``"backward"`` - no normalization
            * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the FFT orthonormal)

            Calling the backward transform (:func:`~torch.fft.ifft`) with the same
            normalization mode will apply an overall normalization of ``1/n`` between
            the two transforms. This is required to make :func:`~torch.fft.ifft`
            the exact inverse.

            Default is ``"backward"`` (no normalization).

    Keyword args:
        {out}

    Example:
    
        >>> t = flow.arange(4)
        >>> t
        tensor([0, 1, 2, 3])
        >>> flow.fft.fft(t)
        tensor([ 6+0j, -2+2j, -2+0j, -2-2j], dtype=oneflow.complex64)

        >>> t = flow.tensor([0.+1.j, 2.+3.j, 4.+5.j, 6.+7.j])
        >>> flow.fft.fft(t)
        tensor([12+16j, -8+0j, -4-4j,  -8j], dtype=oneflow.complex128)
    """
    return flow._C.fft(input, n, dim, norm)


def ifft(input, n=None, dim=-1, norm=None) -> Tensor:
    return flow._C.ifft(input, n, dim, norm)


def fft2(input, s=None, dim=(-2, -1), norm=None) -> Tensor:
    return flow._C.fft2(input, s, dim, norm)


def ifft2(input, s=None, dim=(-2, -1), norm=None) -> Tensor:
    return flow._C.ifft2(input, s, dim, norm)


def fftn(input, s=None, dim=None, norm=None) -> Tensor:
    return flow._C.fftn(input, s, dim, norm)


def ifftn(input, s=None, dim=None, norm=None) -> Tensor:
    return flow._C.ifftn(input, s, dim, norm)


def rfft(input, n=None, dim=-1, norm=None) -> Tensor:
    return flow._C.rfft(input, n, dim, norm)


def irfft(input, n=None, dim=-1, norm=None) -> Tensor:
    return flow._C.irfft(input, n, dim, norm)


def rfft2(input, s=None, dim=(-2, -1), norm=None) -> Tensor:
    return flow._C.rfft2(input, s, dim, norm)


def irfft2(input, s=None, dim=(-2, -1), norm=None) -> Tensor:
    return flow._C.irfft2(input, s, dim, norm)


def rfftn(input, s=None, dim=None, norm=None) -> Tensor:
    return flow._C.rfftn(input, s, dim, norm)


def irfftn(input, s=None, dim=None, norm=None) -> Tensor:
    return flow._C.irfftn(input, s, dim, norm)


def hfft(input, n=None, dim=-1, norm=None) -> Tensor:
    return flow._C.hfft(input, n, dim, norm)


def ihfft(input, n=None, dim=-1, norm=None) -> Tensor:
    return flow._C.ihfft(input, n, dim, norm)


def hfft2(input, s=None, dim=(-2, -1), norm=None) -> Tensor:
    return flow._C.hfft2(input, s, dim, norm)


def ihfft2(input, s=None, dim=(-2, -1), norm=None) -> Tensor:
    return flow._C.ihfft2(input, s, dim, norm)


def hfftn(input, s=None, dim=None, norm=None) -> Tensor:
    return flow._C.hfftn(input, s, dim, norm)


def ihfftn(input, s=None, dim=None, norm=None) -> Tensor:
    return flow._C.ihfftn(input, s, dim, norm)
