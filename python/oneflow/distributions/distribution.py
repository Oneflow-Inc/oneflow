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
import oneflow
import warnings
from typing import Dict, Optional, Any

# NOTE(Liang Depeng): Modified from
#                     https://github.com/pytorch/pytorch/blob/master/torch/distributions/distribution.py

__all__ = ["Distribution"]


class Distribution(object):
    r"""
    Distribution is the abstract base class for probability distributions.
    """

    has_rsample = False
    has_enumerate_support = False
    _validate_args = __debug__

    @staticmethod
    def set_default_validate_args(value):
        """
        Sets whether validation is enabled or disabled.

        The default behavior mimics Python's ``assert`` statement: validation
        is on by default, but is disabled if Python is run in optimized mode
        (via ``python -O``). Validation may be expensive, so you may want to
        disable it once a model is working.

        Args:
            value (bool): Whether to enable validation.
        """
        if value not in [True, False]:
            raise ValueError
        Distribution._validate_args = value

    def __init__(
        self, batch_shape=oneflow.Size(), event_shape=oneflow.Size(), validate_args=None
    ):
        self._batch_shape = batch_shape
        self._event_shape = event_shape
        assert validate_args is None, "only support validate_args=None for now."
        super(Distribution, self).__init__()

    def expand(self, batch_shape, _instance=None):
        """
        Returns a new distribution instance (or populates an existing instance
        provided by a derived class) with batch dimensions expanded to
        `batch_shape`. This method calls :class:`~oneflow.Tensor.expand` on
        the distribution's parameters. As such, this does not allocate new
        memory for the expanded distribution instance. Additionally,
        this does not repeat any args checking or parameter broadcasting in
        `__init__.py`, when an instance is first created.

        Args:
            batch_shape (oneflow.Size): the desired expanded size.
            _instance: new instance provided by subclasses that
                need to override `.expand`.

        Returns:
            New distribution instance with batch dimensions expanded to
            `batch_size`.
        """
        raise NotImplementedError

    @property
    def batch_shape(self):
        """
        Returns the shape over which parameters are batched.
        """
        return self._batch_shape

    @property
    def event_shape(self):
        """
        Returns the shape of a single sample (without batching).
        """
        return self._event_shape

    @property
    def mean(self):
        """
        Returns the mean of the distribution.
        """
        raise NotImplementedError

    @property
    def mode(self):
        """
        Returns the mode of the distribution.
        """
        raise NotImplementedError(f"{self.__class__} does not implement mode")

    @property
    def variance(self):
        """
        Returns the variance of the distribution.
        """
        raise NotImplementedError

    @property
    def stddev(self):
        """
        Returns the standard deviation of the distribution.
        """
        return self.variance.sqrt()

    def sample(self, sample_shape=oneflow.Size()):
        """
        Generates a sample_shape shaped sample or sample_shape shaped batch of
        samples if the distribution parameters are batched.
        """
        with oneflow.no_grad():
            return self.rsample(sample_shape)

    def rsample(self, sample_shape=oneflow.Size()):
        """
        Generates a sample_shape shaped reparameterized sample or sample_shape
        shaped batch of reparameterized samples if the distribution parameters
        are batched.
        """
        raise NotImplementedError

    def sample_n(self, n):
        """
        Generates n samples or n batches of samples if the distribution
        parameters are batched.
        """
        warnings.warn(
            "sample_n will be deprecated. Use .sample((n,)) instead", UserWarning
        )
        return self.sample(oneflow.Size((n,)))

    def log_prob(self, value):
        """
        Returns the log of the probability density/mass function evaluated at
        `value`.

        Args:
            value (Tensor):
        """
        raise NotImplementedError

    def cdf(self, value):
        """
        Returns the cumulative density/mass function evaluated at
        `value`.

        Args:
            value (Tensor):
        """
        raise NotImplementedError

    def icdf(self, value):
        """
        Returns the inverse cumulative density/mass function evaluated at
        `value`.

        Args:
            value (Tensor):
        """
        raise NotImplementedError

    def enumerate_support(self, expand=True):
        """
        Returns tensor containing all values supported by a discrete
        distribution. The result will enumerate over dimension 0, so the shape
        of the result will be `(cardinality,) + batch_shape + event_shape`
        (where `event_shape = ()` for univariate distributions).

        Note that this enumerates over all batched tensors in lock-step
        `[[0, 0], [1, 1], ...]`. With `expand=False`, enumeration happens
        along dim 0, but with the remaining batch dimensions being
        singleton dimensions, `[[0], [1], ..`.

        To iterate over the full Cartesian product use
        `itertools.product(m.enumerate_support())`.

        Args:
            expand (bool): whether to expand the support over the
                batch dims to match the distribution's `batch_shape`.

        Returns:
            Tensor iterating over dimension 0.
        """
        raise NotImplementedError

    def entropy(self):
        """
        Returns entropy of distribution, batched over batch_shape.

        Returns:
            Tensor of shape batch_shape.
        """
        raise NotImplementedError

    def perplexity(self):
        """
        Returns perplexity of distribution, batched over batch_shape.

        Returns:
            Tensor of shape batch_shape.
        """
        return oneflow.exp(self.entropy())

    def _extended_shape(self, sample_shape=oneflow.Size()):
        """
        Returns the size of the sample returned by the distribution, given
        a `sample_shape`. Note, that the batch and event shapes of a distribution
        instance are fixed at the time of construction. If this is empty, the
        returned shape is upcast to (1,).

        Args:
            sample_shape (oneflow.Size): the size of the sample to be drawn.
        """
        if not isinstance(sample_shape, oneflow.Size):
            sample_shape = oneflow.Size(sample_shape)
        return sample_shape + self._batch_shape + self._event_shape

    def _validate_sample(self, value):
        """
        Argument validation for distribution methods such as `log_prob`,
        `cdf` and `icdf`. The rightmost dimensions of a value to be
        scored via these methods must agree with the distribution's batch
        and event shapes.

        Args:
            value (Tensor): the tensor whose log probability is to be
                computed by the `log_prob` method.
        Raises
            ValueError: when the rightmost dimensions of `value` do not match the
                distribution's batch and event shapes.
        """
        if not isinstance(value, oneflow.Tensor):
            raise ValueError("The value argument to log_prob must be a Tensor")

        event_dim_start = len(value.size()) - len(self._event_shape)
        if value.size()[event_dim_start:] != self._event_shape:
            raise ValueError(
                "The right-most size of value must match event_shape: {} vs {}.".format(
                    value.size(), self._event_shape
                )
            )

        actual_shape = value.size()
        expected_shape = self._batch_shape + self._event_shape
        for i, j in zip(reversed(actual_shape), reversed(expected_shape)):
            if i != 1 and j != 1 and i != j:
                raise ValueError(
                    "Value is not broadcastable with batch_shape+event_shape: {} vs {}.".format(
                        actual_shape, expected_shape
                    )
                )
        try:
            support = self.support
        except NotImplementedError:
            warnings.warn(
                f"{self.__class__} does not define `support` to enable "
                + "sample validation. Please initialize the distribution with "
                + "`validate_args=False` to turn off validation."
            )
            return
        assert support is not None
        valid = support.check(value)
        if not valid.all():
            raise ValueError(
                "Expected value argument "
                f"({type(value).__name__} of shape {tuple(value.shape)}) "
                f"to be within the support ({repr(support)}) "
                f"of the distribution {repr(self)}, "
                f"but found invalid values:\n{value}"
            )

    def _get_checked_instance(self, cls, _instance=None):
        if _instance is None and type(self).__init__ != cls.__init__:
            raise NotImplementedError(
                "Subclass {} of {} that defines a custom __init__ method "
                "must also define a custom .expand() method.".format(
                    self.__class__.__name__, cls.__name__
                )
            )
        return self.__new__(type(self)) if _instance is None else _instance

    def __repr__(self):
        return self.__class__.__name__


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
