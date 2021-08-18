
from oneflow._oneflow_internal.autograd import NoGradGuard


class no_grad(NoGradGuard):
    r"""
    Context-manager that disabled gradient calculation.

    Disabling gradient calculation is useful for inference, when you are sure that
    you will not call Tensor.backward(). It will reduce memory consumption for computations
    that would otherwise have requires_grad=True.

    In this mode, the result of every computation will have requires_grad=False, even when
    the inputs have requires_grad=True.

    This context manager is thread local; it will not affect computation in other threads.

    Also functions as a decorator. (Make sure to instantiate with parenthesis.)

    .. code-block:: python

        >>> import oneflow as flow
        >>> x = flow.ones(2, 3, requires_grad=True)
        >>> with flow.no_grad():
        ...     y = x * x
        >>> y.requires_grad
        False
        >>> @flow.no_grad()
        ... def no_grad_func(x):
        ...     return x * x
        >>> y = no_grad_func(x)
        >>> y.requires_grad
        False
    """

    def __call__(self, func):
        def warpper(*args, **kwargs):
            with NoGradGuard():
                return func(*args, **kwargs)

        return warpper


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
