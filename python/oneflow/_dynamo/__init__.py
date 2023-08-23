import warnings

# Reference: https://github.com/pytorch/pytorch/blob/v2.0.1/torch/_dynamo/__init__.py
__all__ = [
    "allow_in_graph",
]

def allow_in_graph(fn):
    """
    """
    if isinstance(fn, (list, tuple)):
        return [allow_in_graph(x) for x in fn]
    assert callable(fn), "allow_in_graph expects a callable"
    warnings.warn(
        "The oneflow._dynamo.allow_in_graph interface is just to align the torch._dynamo.allow_in_graph interface and has no practical significance."
    )
    return fn

