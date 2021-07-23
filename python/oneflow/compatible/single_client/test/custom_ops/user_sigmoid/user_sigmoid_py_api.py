import os
import oneflow.compatible.single_client as flow
from typing import Union, Tuple, List, Optional, Sequence, Callable


def user_sigmoid_forward(x, name: Optional[str] = None):
    return (
        flow.user_op_builder(
            name if name is not None else flow.util.unique_str("UserSigmoidForward_")
        )
        .Op("user_sigmoid_forward")
        .Input("x", [x])
        .Output("y")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


def user_sigmoid_backward(y, dy, name: Optional[str] = None):
    return (
        flow.user_op_builder(
            name if name is not None else flow.util.unique_str("UerSigmoidBackward_")
        )
        .Op("user_sigmoid_backward")
        .Input("y", [y])
        .Input("dy", [dy])
        .Output("dx")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )
