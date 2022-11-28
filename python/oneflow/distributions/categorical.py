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
import oneflow as flow
from oneflow.distributions.distribution import Distribution
from oneflow.distributions.utils import probs_to_logits, logits_to_probs

# NOTE(Liang Depeng): modified from
# https://github.com/pytorch/pytorch/blob/master/torch/distributions/categorical.py

__all__ = ["Categorical"]


class Categorical(Distribution):
    r"""
    Creates a categorical distribution parameterized by either :attr:`probs` or
    :attr:`logits` (but not both).

    .. note::
        It is equivalent to the distribution that :func:`oneflow.multinomial`
        samples from.

    Samples are integers from :math:`\{0, \ldots, K-1\}` where `K` is ``probs.size(-1)``.
    If `probs` is 1-dimensional with length-`K`, each element is the relative probability
    of sampling the class at that index.
    If `probs` is N-dimensional, the first N-1 dimensions are treated as a batch of
    relative probability vectors.

    .. note:: The `probs` argument must be non-negative, finite and have a non-zero sum,
              and it will be normalized to sum to 1 along the last dimension. :attr:`probs`
              will return this normalized value.
              The `logits` argument will be interpreted as unnormalized log probabilities
              and can therefore be any real number. It will likewise be normalized so that
              the resulting probabilities sum to 1 along the last dimension. :attr:`logits`
              will return this normalized value.

    Args:
        probs (Tensor): event probabilities
        logits (Tensor): event log probabilities (unnormalized)

    See also: :func:`oneflow.multinomial`
    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> gen = flow.manual_seed(0)
        >>> m = flow.distributions.categorical.Categorical(flow.tensor([ 0.25, 0.25, 0.25, 0.25 ]))
        >>> m.sample()  # equal probability of 0, 1, 2, 3
        tensor(3, dtype=oneflow.int64)
    """
    has_enumerate_support = True

    def __init__(self, probs=None, logits=None, validate_args=None):
        if (probs is None) == (logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )
        assert validate_args is None

        if probs is not None:
            if probs.dim() < 1:
                raise ValueError("`probs` parameter must be at least one-dimensional.")
            self.probs = probs / probs.sum(-1, keepdim=True)
        else:
            if logits.dim() < 1:
                raise ValueError("`logits` parameter must be at least one-dimensional.")
            self.logits = logits
            # Normalize

            import math

            def logsumexp(t):
                if t.numel() != 0:
                    maxes = flow.max(t, dim=-1, keepdim=True)[0]
                    maxes.masked_fill_(flow.abs(maxes) == math.inf, 0)
                    result = flow.sum(flow.exp(t - maxes), dim=-1, keepdim=True)
                    return flow.log(result) + maxes
                else:
                    return flow.log(flow.sum(t, dim=-1, keepdim=True))

            self.probs = logits_to_probs(logits - logsumexp(logits))

        self._param = self.probs if probs is not None else self.logits
        self._num_events = self._param.size()[-1]
        batch_shape = (
            self._param.size()[:-1] if self._param.ndimension() > 1 else flow.Size()
        )
        super(Categorical, self).__init__(batch_shape, validate_args=validate_args)

    def logits(self):
        return probs_to_logits(self.probs)

    def probs(self):
        return logits_to_probs(self.logits)

    def sample(self, sample_shape=flow.Size()):
        if not isinstance(sample_shape, flow.Size):
            sample_shape = flow.Size(sample_shape)
        probs_2d = self.probs.reshape(-1, self._num_events)
        samples_2d = flow.multinomial(probs_2d, sample_shape.numel(), True).T
        return samples_2d.reshape(self._extended_shape(sample_shape))


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
