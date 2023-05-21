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
    
    Note:
    The Fourier domain representation of any real signal satisfies the
    Hermitian property: `X[i] = conj(X[-i])`. This function always returns both
    the positive and negative frequency terms even though, for real inputs, the
    negative frequencies are redundant. :func:`oneflow.fft.rfft` returns the
    more compact one-sided representation where only the positive frequencies
    are returned.

    Args:
        input (Tensor): the input tensor
        n (int, optional): Signal length. If given, the input will either be zero-padded
            or trimmed to this length before computing the FFT.
        dim (int, optional): The dimension along which to take the one dimensional FFT.
        norm (str, optional): Normalization mode. For the forward transform
            (:func:`oneflow.fft.fft`), these correspond to:

            * ``"forward"`` - normalize by ``1/n``
            * ``"backward"`` - no normalization
            * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the FFT orthonormal)

            Calling the backward transform (:func:`oneflow.fft.ifft`) with the same
            normalization mode will apply an overall normalization of ``1/n`` between
            the two transforms. This is required to make :func:`oneflow.fft.ifft`
            the exact inverse.

            Default is ``"backward"`` (no normalization).

    Example:
    
        >>> t = oneflow.arange(4)
        >>> t
        tensor([0, 1, 2, 3])
        >>> oneflow.fft.fft(t)
        tensor([ 6+0j, -2+2j, -2+0j, -2-2j], dtype=oneflow.complex64)

        >>> t = oneflow.tensor([0.+1.j, 2.+3.j, 4.+5.j, 6.+7.j])
        >>> oneflow.fft.fft(t)
        tensor([12+16j, -8+0j, -4-4j,  -8j], dtype=oneflow.complex128)
    """
    if n is None:
        n = -1
    return flow._C.fft(input, n, dim, norm)


def ifft(input, n=None, dim=-1, norm=None) -> Tensor:
    r"""

    Computes the one dimensional inverse discrete Fourier transform of :attr:`input`.

    Args:
        input (Tensor): the input tensor
        n (int, optional): Signal length. If given, the input will either be zero-padded
            or trimmed to this length before computing the IFFT.
        dim (int, optional): The dimension along which to take the one dimensional IFFT.
        norm (str, optional): Normalization mode. For the backward transform
            (:func:`oneflow.fft.ifft`), these correspond to:

            * ``"forward"`` - no normalization
            * ``"backward"`` - normalize by ``1/n``
            * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the IFFT orthonormal)

            Calling the forward transform (:func:`~oneflow.fft.fft`) with the same
            normalization mode will apply an overall normalization of ``1/n`` between
            the two transforms. This is required to make :func:`oneflow.fft.ifft`
            the exact inverse.

            Default is ``"backward"`` (normalize by ``1/n``).

    Example:

        >>> t = oneflow.tensor([ 6.+0.j, -2.+2.j, -2.+0.j, -2.-2.j])
        >>> oneflow.fft.ifft(t)
        tensor([0j, (1+0j), (2+0j), (3+0j)], dtype=oneflow.complex128)
    """
    if n is None:
        n = -1
    return flow._C.ifft(input, n, dim, norm)


def fft2(input, s=None, dim=(-2, -1), norm=None) -> Tensor:
    r"""

    Computes the 2 dimensional discrete Fourier transform of :attr:`input`.
    Equivalent to :func:`~oneflow.fft.fftn` but FFTs only the last two dimensions by default.

    Note:
        The Fourier domain representation of any real signal satisfies the
        Hermitian property: ``X[i, j] = conj(X[-i, -j])``. This
        function always returns all positive and negative frequency terms even
        though, for real inputs, half of these values are redundant.
        :func:`~oneflow.fft.rfft2` returns the more compact one-sided representation
        where only the positive frequencies of the last dimension are returned.

    Args:
        input (Tensor): the input tensor
        s (Tuple[int], optional): Signal size in the transformed dimensions.
            If given, each dimension ``dim[i]`` will either be zero-padded or
            trimmed to the length ``s[i]`` before computing the FFT.
            If a length ``-1`` is specified, no padding is done in that dimension.
            Default: ``s = [input.size(d) for d in dim]``
        dim (Tuple[int], optional): Dimensions to be transformed.
            Default: last two dimensions.
        norm (str, optional): Normalization mode. For the forward transform
            (:func:`oneflow.fft.fft2`), these correspond to:

            * ``"forward"`` - normalize by ``1/n``
            * ``"backward"`` - no normalization
            * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the FFT orthonormal)

            Where ``n = prod(s)`` is the logical FFT size.
            Calling the backward transform (:func:`oneflow.fft.ifft2`) with the same
            normalization mode will apply an overall normalization of ``1/n``
            between the two transforms. This is required to make
            :func:`~oneflow.fft.ifft2` the exact inverse.

            Default is ``"backward"`` (no normalization).

    """
    return flow._C.fft2(input, s, dim, norm)


def ifft2(input, s=None, dim=(-2, -1), norm=None) -> Tensor:
    r"""

    Computes the 2 dimensional inverse discrete Fourier transform of :attr:`input`.
    Equivalent to :func:`oneflow.fft.ifftn` but IFFTs only the last two dimensions by default.

    Args:
        input (Tensor): the input tensor
        s (Tuple[int], optional): Signal size in the transformed dimensions.
            If given, each dimension ``dim[i]`` will either be zero-padded or
            trimmed to the length ``s[i]`` before computing the IFFT.
            If a length ``-1`` is specified, no padding is done in that dimension.
            Default: ``s = [input.size(d) for d in dim]``
        dim (Tuple[int], optional): Dimensions to be transformed.
            Default: last two dimensions.
        norm (str, optional): Normalization mode. For the backward transform
            (:func:`oneflow.fft.ifft2`), these correspond to:

            * ``"forward"`` - no normalization
            * ``"backward"`` - normalize by ``1/n``
            * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the IFFT orthonormal)

            Where ``n = prod(s)`` is the logical IFFT size.
            Calling the forward transform (:func:`oneflow.fft.fft2`) with the same
            normalization mode will apply an overall normalization of ``1/n`` between
            the two transforms. This is required to make :func:`oneflow.fft.ifft2`
            the exact inverse.

            Default is ``"backward"`` (normalize by ``1/n``).


    """
    return flow._C.ifft2(input, s, dim, norm)


def fftn(input, s=None, dim=None, norm=None) -> Tensor:
    r"""

    Computes the N dimensional discrete Fourier transform of :attr:`input`.

    Note:
        The Fourier domain representation of any real signal satisfies the
        Hermitian property: ``X[i_1, ..., i_n] = conj(X[-i_1, ..., -i_n])``. This
        function always returns all positive and negative frequency terms even
        though, for real inputs, half of these values are redundant.
        :func:`oneflow.fft.rfftn` returns the more compact one-sided representation
        where only the positive frequencies of the last dimension are returned.

    Args:
        input (Tensor): the input tensor
        s (Tuple[int], optional): Signal size in the transformed dimensions.
            If given, each dimension ``dim[i]`` will either be zero-padded or
            trimmed to the length ``s[i]`` before computing the FFT.
            If a length ``-1`` is specified, no padding is done in that dimension.
            Default: ``s = [input.size(d) for d in dim]``
        dim (Tuple[int], optional): Dimensions to be transformed.
            Default: all dimensions, or the last ``len(s)`` dimensions if :attr:`s` is given.
        norm (str, optional): Normalization mode. For the forward transform
            (:func:`oneflow.fft.fftn`), these correspond to:

            * ``"forward"`` - normalize by ``1/n``
            * ``"backward"`` - no normalization
            * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the FFT orthonormal)

            Where ``n = prod(s)`` is the logical FFT size.
            Calling the backward transform (:func:`oneflow.fft.ifftn`) with the same
            normalization mode will apply an overall normalization of ``1/n``
            between the two transforms. This is required to make
            :func:`oneflow.fft.ifftn` the exact inverse.

            Default is ``"backward"`` (no normalization).

    """
    return flow._C.fftn(input, s, dim, norm)


def ifftn(input, s=None, dim=None, norm=None) -> Tensor:
    r"""

    Computes the N dimensional inverse discrete Fourier transform of :attr:`input`.

    Args:
        input (Tensor): the input tensor
        s (Tuple[int], optional): Signal size in the transformed dimensions.
            If given, each dimension ``dim[i]`` will either be zero-padded or
            trimmed to the length ``s[i]`` before computing the IFFT.
            If a length ``-1`` is specified, no padding is done in that dimension.
            Default: ``s = [input.size(d) for d in dim]``
        dim (Tuple[int], optional): Dimensions to be transformed.
            Default: all dimensions, or the last ``len(s)`` dimensions if :attr:`s` is given.
        norm (str, optional): Normalization mode. For the backward transform
            (:func:`oneflow.fft.ifftn`), these correspond to:

            * ``"forward"`` - no normalization
            * ``"backward"`` - normalize by ``1/n``
            * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the IFFT orthonormal)

            Where ``n = prod(s)`` is the logical IFFT size.
            Calling the forward transform (:func:`oneflow.fft.fftn`) with the same
            normalization mode will apply an overall normalization of ``1/n`` between
            the two transforms. This is required to make :func:`oneflow.fft.ifftn`
            the exact inverse.

            Default is ``"backward"`` (normalize by ``1/n``).

    """
    return flow._C.ifftn(input, s, dim, norm)


def rfft(input, n=None, dim=-1, norm=None) -> Tensor:
    r"""

    Computes the one dimensional Fourier transform of real-valued :attr:`input`.

    The FFT of a real signal is Hermitian-symmetric, ``X[i] = conj(X[-i])`` so
    the output contains only the positive frequencies below the Nyquist frequency.
    To compute the full output, use :func:`oneflow.fft.fft`

    Args:
        input (Tensor): the real input tensor
        n (int, optional): Signal length. If given, the input will either be zero-padded
            or trimmed to this length before computing the real FFT.
        dim (int, optional): The dimension along which to take the one dimensional real FFT.
        norm (str, optional): Normalization mode. For the forward transform
            (:func:`oneflow.fft.rfft`), these correspond to:

            * ``"forward"`` - normalize by ``1/n``
            * ``"backward"`` - no normalization
            * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the FFT orthonormal)

            Calling the backward transform (:func:`oneflow.fft.irfft`) with the same
            normalization mode will apply an overall normalization of ``1/n`` between
            the two transforms. This is required to make :func:`oneflow.fft.irfft`
            the exact inverse.

            Default is ``"backward"`` (no normalization).

    Example:

        >>> t = oneflow.arange(4)
        >>> t
        tensor([0, 1, 2, 3], dtype=oneflow.int64)
        >>> oneflow.fft.rfft(t)
        tensor([ (6+0j), (-2+2j), (-2+0j)], dtype=oneflow.complex64)

        Compare against the full output from :func:`oneflow.fft.fft`:

        >>> oneflow.fft.fft(t)
        tensor([ (6+0j), (-2+2j), (-2+0j), (-2-2j)], dtype=oneflow.complex64)

        Notice that the symmetric element ``T[-1] == T[1].conj()`` is omitted.
        At the Nyquist frequency ``T[-2] == T[2]`` is it's own symmetric pair,
        and therefore must always be real-valued.
    """

    if n is None:
        n = -1
    return flow._C.rfft(input, n, dim, norm)


def irfft(input, n=None, dim=-1, norm=None) -> Tensor:
    r"""

    Computes the inverse of :func:`oneflow.fft.rfft`.

    :attr:`input` is interpreted as a one-sided Hermitian signal in the Fourier
    domain, as produced by :func:`oneflow.fft.rfft`. By the Hermitian property, the
    output will be real-valued.

    Note:
        Some input frequencies must be real-valued to satisfy the Hermitian
        property. In these cases the imaginary component will be ignored.
        For example, any imaginary component in the zero-frequency term cannot
        be represented in a real output and so will always be ignored.

    Note:
        The correct interpretation of the Hermitian input depends on the length of
        the original data, as given by :attr:`n`. This is because each input shape
        could correspond to either an odd or even length signal. By default, the
        signal is assumed to be even length and odd signals will not round-trip
        properly. So, it is recommended to always pass the signal length :attr:`n`.

    Args:
        input (Tensor): the input tensor representing a half-Hermitian signal
        n (int, optional): Output signal length. This determines the length of the
            output signal. If given, the input will either be zero-padded or trimmed to this
            length before computing the real IFFT.
            Defaults to even output: ``n=2*(input.size(dim) - 1)``.
        dim (int, optional): The dimension along which to take the one dimensional real IFFT.
        norm (str, optional): Normalization mode. For the backward transform
            (:func:`oneflow.fft.irfft`), these correspond to:

            * ``"forward"`` - no normalization
            * ``"backward"`` - normalize by ``1/n``
            * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the real IFFT orthonormal)

            Calling the forward transform (:func:`oneflow.fft.rfft`) with the same
            normalization mode will apply an overall normalization of ``1/n`` between
            the two transforms. This is required to make :func:`oneflow.fft.irfft`
            the exact inverse.

            Default is ``"backward"`` (normalize by ``1/n``).


    """

    if n is None:
        n = -1
    return flow._C.irfft(input, n, dim, norm)


def rfft2(input, s=None, dim=(-2, -1), norm=None) -> Tensor:
    r"""

    Computes the 2-dimensional discrete Fourier transform of real :attr:`input`.
    Equivalent to :func:`oneflow.fft.rfftn` but FFTs only the last two dimensions by default.

    The FFT of a real signal is Hermitian-symmetric, ``X[i, j] = conj(X[-i, -j])``,
    so the full :func:`oneflow.fft.fft2` output contains redundant information.
    :func:`oneflow.fft.rfft2` instead omits the negative frequencies in the last
    dimension.

    Args:
        input (Tensor): the input tensor
        s (Tuple[int], optional): Signal size in the transformed dimensions.
            If given, each dimension ``dim[i]`` will either be zero-padded or
            trimmed to the length ``s[i]`` before computing the real FFT.
            If a length ``-1`` is specified, no padding is done in that dimension.
            Default: ``s = [input.size(d) for d in dim]``
        dim (Tuple[int], optional): Dimensions to be transformed.
            Default: last two dimensions.
        norm (str, optional): Normalization mode. For the forward transform
            (:func:`oneflow.fft.rfft2`), these correspond to:

            * ``"forward"`` - normalize by ``1/n``
            * ``"backward"`` - no normalization
            * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the real FFT orthonormal)

            Where ``n = prod(s)`` is the logical FFT size.
            Calling the backward transform (:func:`oneflow.fft.irfft2`) with the same
            normalization mode will apply an overall normalization of ``1/n`` between
            the two transforms. This is required to make :func:`oneflow.fft.irfft2`
            the exact inverse.

            Default is ``"backward"`` (no normalization).

    """

    return flow._C.rfft2(input, s, dim, norm)


def irfft2(input, s=None, dim=(-2, -1), norm=None) -> Tensor:
    r"""

    Computes the inverse of :func:`oneflow.fft.rfft2`.
    Equivalent to :func:`oneflow.fft.irfftn` but IFFTs only the last two dimensions by default.

    :attr:`input` is interpreted as a one-sided Hermitian signal in the Fourier
    domain, as produced by :func:`oneflow.fft.rfft2`. By the Hermitian property, the
    output will be real-valued.

    Note:
        Some input frequencies must be real-valued to satisfy the Hermitian
        property. In these cases the imaginary component will be ignored.
        For example, any imaginary component in the zero-frequency term cannot
        be represented in a real output and so will always be ignored.

    Note:
        The correct interpretation of the Hermitian input depends on the length of
        the original data, as given by :attr:`s`. This is because each input shape
        could correspond to either an odd or even length signal. By default, the
        signal is assumed to be even length and odd signals will not round-trip
        properly. So, it is recommended to always pass the signal shape :attr:`s`.

    Args:
        input (Tensor): the input tensor
        s (Tuple[int], optional): Signal size in the transformed dimensions.
            If given, each dimension ``dim[i]`` will either be zero-padded or
            trimmed to the length ``s[i]`` before computing the real FFT.
            If a length ``-1`` is specified, no padding is done in that dimension.
            Defaults to even output in the last dimension:
            ``s[-1] = 2*(input.size(dim[-1]) - 1)``.
        dim (Tuple[int], optional): Dimensions to be transformed.
            The last dimension must be the half-Hermitian compressed dimension.
            Default: last two dimensions.
        norm (str, optional): Normalization mode. For the backward transform
            (:func:`oneflow.fft.irfft2`), these correspond to:

            * ``"forward"`` - no normalization
            * ``"backward"`` - normalize by ``1/n``
            * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the real IFFT orthonormal)

            Where ``n = prod(s)`` is the logical IFFT size.
            Calling the forward transform (:func:`oneflow.fft.rfft2`) with the same
            normalization mode will apply an overall normalization of ``1/n`` between
            the two transforms. This is required to make :func:`oneflow.fft.irfft2`
            the exact inverse.

            Default is ``"backward"`` (normalize by ``1/n``).


    """
    return flow._C.irfft2(input, s, dim, norm)


def rfftn(input, s=None, dim=None, norm=None) -> Tensor:
    r"""

    Computes the N-dimensional discrete Fourier transform of real :attr:`input`.

    The FFT of a real signal is Hermitian-symmetric,
    ``X[i_1, ..., i_n] = conj(X[-i_1, ..., -i_n])`` so the full
    :func:`oneflow.fft.fftn` output contains redundant information.
    :func:`oneflow.fft.rfftn` instead omits the negative frequencies in the
    last dimension.

    Args:
        input (Tensor): the input tensor
        s (Tuple[int], optional): Signal size in the transformed dimensions.
            If given, each dimension ``dim[i]`` will either be zero-padded or
            trimmed to the length ``s[i]`` before computing the real FFT.
            If a length ``-1`` is specified, no padding is done in that dimension.
            Default: ``s = [input.size(d) for d in dim]``
        dim (Tuple[int], optional): Dimensions to be transformed.
            Default: all dimensions, or the last ``len(s)`` dimensions if :attr:`s` is given.
        norm (str, optional): Normalization mode. For the forward transform
            (:func:`oneflow.fft.rfftn`), these correspond to:

            * ``"forward"`` - normalize by ``1/n``
            * ``"backward"`` - no normalization
            * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the real FFT orthonormal)

            Where ``n = prod(s)`` is the logical FFT size.
            Calling the backward transform (:func:`oneflow.fft.irfftn`) with the same
            normalization mode will apply an overall normalization of ``1/n`` between
            the two transforms. This is required to make :func:`oneflow.fft.irfftn`
            the exact inverse.

            Default is ``"backward"`` (no normalization).

    """

    return flow._C.rfftn(input, s, dim, norm)


def irfftn(input, s=None, dim=None, norm=None) -> Tensor:
    r"""

    Computes the inverse of :func:`oneflow.fft.rfftn`.

    :attr:`input` is interpreted as a one-sided Hermitian signal in the Fourier
    domain, as produced by :func:`oneflow.fft.rfftn`. By the Hermitian property, the
    output will be real-valued.

    Note:
        Some input frequencies must be real-valued to satisfy the Hermitian
        property. In these cases the imaginary component will be ignored.
        For example, any imaginary component in the zero-frequency term cannot
        be represented in a real output and so will always be ignored.

    Note:
        The correct interpretation of the Hermitian input depends on the length of
        the original data, as given by :attr:`s`. This is because each input shape
        could correspond to either an odd or even length signal. By default, the
        signal is assumed to be even length and odd signals will not round-trip
        properly. So, it is recommended to always pass the signal shape :attr:`s`.

    Args:
        input (Tensor): the input tensor
        s (Tuple[int], optional): Signal size in the transformed dimensions.
            If given, each dimension ``dim[i]`` will either be zero-padded or
            trimmed to the length ``s[i]`` before computing the real FFT.
            If a length ``-1`` is specified, no padding is done in that dimension.
            Defaults to even output in the last dimension:
            ``s[-1] = 2*(input.size(dim[-1]) - 1)``.
        dim (Tuple[int], optional): Dimensions to be transformed.
            The last dimension must be the half-Hermitian compressed dimension.
            Default: all dimensions, or the last ``len(s)`` dimensions if :attr:`s` is given.
        norm (str, optional): Normalization mode. For the backward transform
            (:func:`oneflow.fft.irfftn`), these correspond to:

            * ``"forward"`` - no normalization
            * ``"backward"`` - normalize by ``1/n``
            * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the real IFFT orthonormal)

            Where ``n = prod(s)`` is the logical IFFT size.
            Calling the forward transform (:func:`oneflow.fft.rfftn`) with the same
            normalization mode will apply an overall normalization of ``1/n`` between
            the two transforms. This is required to make :func:`oneflow.fft.irfftn`
            the exact inverse.

            Default is ``"backward"`` (normalize by ``1/n``).

    """
    return flow._C.irfftn(input, s, dim, norm)


def hfft(input, n=None, dim=-1, norm=None) -> Tensor:
    r"""
    hfft(input, n=None, dim=-1, norm=None, *, out=None) -> Tensor

    Computes the one dimensional discrete Fourier transform of a Hermitian
    symmetric :attr:`input` signal.

    Note:

        :func:`oneflow.fft.hfft`/:func:`oneflow.fft.ihfft` are analogous to
        :func:`oneflow.fft.rfft`/:func:`oneflow.fft.irfft`. The real FFT expects
        a real signal in the time-domain and gives a Hermitian symmetry in the
        frequency-domain. The Hermitian FFT is the opposite; Hermitian symmetric in
        the time-domain and real-valued in the frequency-domain. For this reason,
        special care needs to be taken with the length argument :attr:`n`, in the
        same way as with :func:`oneflow.fft.irfft`.

    Note:
        Because the signal is Hermitian in the time-domain, the result will be
        real in the frequency domain. Note that some input frequencies must be
        real-valued to satisfy the Hermitian property. In these cases the imaginary
        component will be ignored. For example, any imaginary component in
        ``input[0]`` would result in one or more complex frequency terms which
        cannot be represented in a real output and so will always be ignored.

    Note:
        The correct interpretation of the Hermitian input depends on the length of
        the original data, as given by :attr:`n`. This is because each input shape
        could correspond to either an odd or even length signal. By default, the
        signal is assumed to be even length and odd signals will not round-trip
        properly. So, it is recommended to always pass the signal length :attr:`n`.

    Args:
        input (Tensor): the input tensor representing a half-Hermitian signal
        n (int, optional): Output signal length. This determines the length of the
            real output. If given, the input will either be zero-padded or trimmed to this
            length before computing the Hermitian FFT.
            Defaults to even output: ``n=2*(input.size(dim) - 1)``.
        dim (int, optional): The dimension along which to take the one dimensional Hermitian FFT.
        norm (str, optional): Normalization mode. For the forward transform
            (:func:`oneflow.fft.hfft`), these correspond to:

            * ``"forward"`` - normalize by ``1/n``
            * ``"backward"`` - no normalization
            * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the Hermitian FFT orthonormal)

            Calling the backward transform (:func:`oneflow.fft.ihfft`) with the same
            normalization mode will apply an overall normalization of ``1/n`` between
            the two transforms. This is required to make :func:`oneflow.fft.ihfft`
            the exact inverse.

            Default is ``"backward"`` (no normalization).

    Example:

        Taking a real-valued frequency signal and bringing it into the time domain
        gives Hermitian symmetric output:

        >>> t = oneflow.linspace(0, 1, 5)
        >>> t
        tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000], dtype=oneflow.float32)
        >>> T = oneflow.fft.ifft(t)
        >>> T
        tensor([ (0.5000-0.0000j), (-0.1250-0.1720j), (-0.1250-0.0406j), (-0.1250+0.0406j),
                (-0.1250+0.1720j)], dtype=oneflow.complex64)
        
        Note that ``T[1] == T[-1].conj()`` and ``T[2] == T[-2].conj()`` is
        redundant. We can thus compute the forward transform without considering
        negative frequencies:

        >>> oneflow.fft.hfft(T[:3], n=5)
        tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000], dtype=oneflow.float32)

        Like with :func:`oneflow.fft.irfft`, the output length must be given in order
        to recover an even length output:

        >>> oneflow.fft.hfft(T[:3])
        tensor([0.1250, 0.2809, 0.6250, 0.9691], dtype=oneflow.float32)
    """

    if n is None:
        n = -1
    return flow._C.hfft(input, n, dim, norm)


def ihfft(input, n=None, dim=-1, norm=None) -> Tensor:
    r"""
    
    Computes the inverse of :func:`oneflow.fft.hfft`.

    :attr:`input` must be a real-valued signal, interpreted in the Fourier domain.
    The IFFT of a real signal is Hermitian-symmetric, ``X[i] = conj(X[-i])``.
    :func:`oneflow.fft.ihfft` represents this in the one-sided form where only the
    positive frequencies below the Nyquist frequency are included. To compute the
    full output, use :func:`oneflow.fft.ifft`.


    Args:
        input (Tensor): the real input tensor
        n (int, optional): Signal length. If given, the input will either be zero-padded
            or trimmed to this length before computing the Hermitian IFFT.
        dim (int, optional): The dimension along which to take the one dimensional Hermitian IFFT.
        norm (str, optional): Normalization mode. For the backward transform
            (:func:`oneflow.fft.ihfft`), these correspond to:

            * ``"forward"`` - no normalization
            * ``"backward"`` - normalize by ``1/n``
            * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the IFFT orthonormal)

            Calling the forward transform (:func:`oneflow.fft.hfft`) with the same
            normalization mode will apply an overall normalization of ``1/n`` between
            the two transforms. This is required to make :func:`oneflow.fft.ihfft`
            the exact inverse.

            Default is ``"backward"`` (normalize by ``1/n``).

    Example:

        >>> t = oneflow.arange(5)
        >>> t
        tensor([0, 1, 2, 3, 4], dtype=oneflow.int64)
        >>> oneflow.fft.ihfft(t)
        tensor([ (2.0000-0.0000j), (-0.5000-0.6882j), (-0.5000-0.1625j)], dtype=oneflow.complex64)
        
        Compare against the full output from :func:`oneflow.fft.ifft`:

        >>> oneflow.fft.ifft(t)
        tensor([ 2.0000-0.0000j, -0.5000-0.6882j, -0.5000-0.1625j, -0.5000+0.1625j,
                -0.5000+0.6882j])
        tensor([ (2.0000-0.0000j), (-0.5000-0.6882j), (-0.5000-0.1625j), (-0.5000+0.1625j),
                (-0.5000+0.6882j)], dtype=oneflow.complex64)
    """
    if n is None:
        n = -1
    return flow._C.ihfft(input, n, dim, norm)


def hfft2(input, s=None, dim=(-2, -1), norm=None) -> Tensor:
    r"""

    Computes the 2-dimensional discrete Fourier transform of a Hermitian symmetric
    :attr:`input` signal. Equivalent to :func:`oneflow.fft.hfftn` but only
    transforms the last two dimensions by default.

    :attr:`input` is interpreted as a one-sided Hermitian signal in the time
    domain. By the Hermitian property, the Fourier transform will be real-valued.

    Args:
        input (Tensor): the input tensor
        s (Tuple[int], optional): Signal size in the transformed dimensions.
            If given, each dimension ``dim[i]`` will either be zero-padded or
            trimmed to the length ``s[i]`` before computing the Hermitian FFT.
            If a length ``-1`` is specified, no padding is done in that dimension.
            Defaults to even output in the last dimension:
            ``s[-1] = 2*(input.size(dim[-1]) - 1)``.
        dim (Tuple[int], optional): Dimensions to be transformed.
            The last dimension must be the half-Hermitian compressed dimension.
            Default: last two dimensions.
        norm (str, optional): Normalization mode. For the forward transform
            (:func:`oneflow.fft.hfft2`), these correspond to:

            * ``"forward"`` - normalize by ``1/n``
            * ``"backward"`` - no normalization
            * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the Hermitian FFT orthonormal)

            Where ``n = prod(s)`` is the logical FFT size.
            Calling the backward transform (:func:`oneflow.fft.ihfft2`) with the same
            normalization mode will apply an overall normalization of ``1/n`` between
            the two transforms. This is required to make :func:`oneflow.fft.ihfft2`
            the exact inverse.

            Default is ``"backward"`` (no normalization).


    Example:

        Starting from a real frequency-space signal, we can generate a
        Hermitian-symmetric time-domain signal:
        >>> T = oneflow.rand(10, 9)
        >>> t = oneflow.fft.ihfft2(T)

        Without specifying the output length to :func:`oneflow.fft.hfftn`, the
        output will not round-trip properly because the input is odd-length in the
        last dimension:

        >>> oneflow.fft.hfft2(t).size()
        oneflow.Size([10, 10])

        So, it is recommended to always pass the signal shape :attr:`s`.

        >>> roundtrip = oneflow.fft.hfft2(t, T.size())
        >>> roundtrip.size()
        oneflow.Size([10, 9])
        >>> oneflow.allclose(roundtrip, T)
        True

    """
    return flow._C.hfft2(input, s, dim, norm)


def ihfft2(input, s=None, dim=(-2, -1), norm=None) -> Tensor:
    r"""

    Computes the 2-dimensional inverse discrete Fourier transform of real
    :attr:`input`. Equivalent to :func:`oneflow.fft.ihfftn` but transforms only the
    two last dimensions by default.

    Args:
        input (Tensor): the input tensor
        s (Tuple[int], optional): Signal size in the transformed dimensions.
            If given, each dimension ``dim[i]`` will either be zero-padded or
            trimmed to the length ``s[i]`` before computing the Hermitian IFFT.
            If a length ``-1`` is specified, no padding is done in that dimension.
            Default: ``s = [input.size(d) for d in dim]``
        dim (Tuple[int], optional): Dimensions to be transformed.
            Default: last two dimensions.
        norm (str, optional): Normalization mode. For the backward transform
            (:func:`oneflow.fft.ihfft2`), these correspond to:

            * ``"forward"`` - no normalization
            * ``"backward"`` - normalize by ``1/n``
            * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the Hermitian IFFT orthonormal)

            Where ``n = prod(s)`` is the logical IFFT size.
            Calling the forward transform (:func:`oneflow.fft.hfft2`) with the same
            normalization mode will apply an overall normalization of ``1/n`` between
            the two transforms. This is required to make :func:`oneflow.fft.ihfft2`
            the exact inverse.

            Default is ``"backward"`` (normalize by ``1/n``).

    """
    return flow._C.ihfft2(input, s, dim, norm)


def hfftn(input, s=None, dim=None, norm=None) -> Tensor:
    r"""

    Computes the n-dimensional discrete Fourier transform of a Hermitian symmetric
    :attr:`input` signal.

    :attr:`input` is interpreted as a one-sided Hermitian signal in the time
    domain. By the Hermitian property, the Fourier transform will be real-valued.

    Note:
        :func:`oneflow.fft.hfftn`/:func:`oneflow.fft.ihfftn` are analogous to
        :func:`oneflow.fft.rfftn`/:func:`oneflow.fft.irfftn`. The real FFT expects
        a real signal in the time-domain and gives Hermitian symmetry in the
        frequency-domain. The Hermitian FFT is the opposite; Hermitian symmetric in
        the time-domain and real-valued in the frequency-domain. For this reason,
        special care needs to be taken with the shape argument :attr:`s`, in the
        same way as with :func:`oneflow.fft.irfftn`.

    Note:
        Some input frequencies must be real-valued to satisfy the Hermitian
        property. In these cases the imaginary component will be ignored.
        For example, any imaginary component in the zero-frequency term cannot
        be represented in a real output and so will always be ignored.

    Note:
        The correct interpretation of the Hermitian input depends on the length of
        the original data, as given by :attr:`s`. This is because each input shape
        could correspond to either an odd or even length signal. By default, the
        signal is assumed to be even length and odd signals will not round-trip
        properly. It is recommended to always pass the signal shape :attr:`s`.


    Args:
        input (Tensor): the input tensor
        s (Tuple[int], optional): Signal size in the transformed dimensions.
            If given, each dimension ``dim[i]`` will either be zero-padded or
            trimmed to the length ``s[i]`` before computing the real FFT.
            If a length ``-1`` is specified, no padding is done in that dimension.
            Defaults to even output in the last dimension:
            ``s[-1] = 2*(input.size(dim[-1]) - 1)``.
        dim (Tuple[int], optional): Dimensions to be transformed.
            The last dimension must be the half-Hermitian compressed dimension.
            Default: all dimensions, or the last ``len(s)`` dimensions if :attr:`s` is given.
        norm (str, optional): Normalization mode. For the forward transform
            (:func:`oneflow.fft.hfftn`), these correspond to:

            * ``"forward"`` - normalize by ``1/n``
            * ``"backward"`` - no normalization
            * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the Hermitian FFT orthonormal)

            Where ``n = prod(s)`` is the logical FFT size.
            Calling the backward transform (:func:`oneflow.fft.ihfftn`) with the same
            normalization mode will apply an overall normalization of ``1/n`` between
            the two transforms. This is required to make :func:`oneflow.fft.ihfftn`
            the exact inverse.

            Default is ``"backward"`` (no normalization).

    """
    return flow._C.hfftn(input, s, dim, norm)


def ihfftn(input, s=None, dim=None, norm=None) -> Tensor:
    r"""

    Computes the N-dimensional inverse discrete Fourier transform of real :attr:`input`.

    :attr:`input` must be a real-valued signal, interpreted in the Fourier domain.
    The n-dimensional IFFT of a real signal is Hermitian-symmetric,
    ``X[i, j, ...] = conj(X[-i, -j, ...])``. :func:`oneflow.fft.ihfftn` represents
    this in the one-sided form where only the positive frequencies below the
    Nyquist frequency are included in the last signal dimension. To compute the
    full output, use :func:`oneflow.fft.ifftn`.

    Args:
        input (Tensor): the input tensor
        s (Tuple[int], optional): Signal size in the transformed dimensions.
            If given, each dimension ``dim[i]`` will either be zero-padded or
            trimmed to the length ``s[i]`` before computing the Hermitian IFFT.
            If a length ``-1`` is specified, no padding is done in that dimension.
            Default: ``s = [input.size(d) for d in dim]``
        dim (Tuple[int], optional): Dimensions to be transformed.
            Default: all dimensions, or the last ``len(s)`` dimensions if :attr:`s` is given.
        norm (str, optional): Normalization mode. For the backward transform
            (:func:`oneflow.fft.ihfftn`), these correspond to:

            * ``"forward"`` - no normalization
            * ``"backward"`` - normalize by ``1/n``
            * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the Hermitian IFFT orthonormal)

            Where ``n = prod(s)`` is the logical IFFT size.
            Calling the forward transform (:func:`oneflow.fft.hfftn`) with the same
            normalization mode will apply an overall normalization of ``1/n`` between
            the two transforms. This is required to make :func:`oneflow.fft.ihfftn`
            the exact inverse.

            Default is ``"backward"`` (normalize by ``1/n``).

    """
    return flow._C.ihfftn(input, s, dim, norm)
