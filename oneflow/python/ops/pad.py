from __future__ import absolute_import

import oneflow
from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.framework.id_util as id_util

@oneflow_export("pad")
def pad(x, paddings, constant_value=0, name=None):
    padding_before = []
    padding_after = []
    if isinstance(paddings, (list, tuple)):
        assert len(paddings) == len(x.static_shape), ValueError(
            "paddings must be the same size of input dims"
        )
        for p in paddings:
            assert isinstance(p, (list, tuple)) and len(p) == 2, ValueError(
                "the elem of paddings must be a tuple or a list with length of 2"
            )
            padding_before.append(p[0])
            padding_after.append(p[1])
    else:
        raise ValueError("paddings must be a tuple or a list.")
    return (
        oneflow.user_op_builder(name if name is not None else id_util.UniqueStr("Pad_"))
        .Op("pad")
        .Input("x", [x])
        .Output("y")
        .SetAttr("padding_before", padding_before, "AttrTypeListInt64")
        .SetAttr("padding_after", padding_after, "AttrTypeListInt64")
        .SetAttr("floating_constant_value", float(constant_value), "AttrTypeDouble")
        .SetAttr("integral_constant_value", int(constant_value), "AttrTypeInt64")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("pad_grad")
def pad_grad(x, paddings, constant_value=0, name=None):
    padding_before = []
    padding_after = []
    if isinstance(paddings, (list, tuple)):
        assert len(paddings) == len(x.static_shape), ValueError(
            "paddings must be the same size of input dims"
        )
        for p in paddings:
            assert isinstance(p, (list, tuple)) and len(p) == 2, ValueError(
                "the elem of paddings must be a tuple or a list with length of 2"
            )
            padding_before.append(p[0])
            padding_after.append(p[1])
    else:
        raise ValueError("paddings must be a tuple or a list.")
    return (
        oneflow.user_op_builder(name if name is not None else id_util.UniqueStr("PadGrad_"))
        .Op("pad_grad")
        .Input("dy", [x])
        .Output("dx")
        .SetAttr("padding_before", padding_before, "AttrTypeListInt64")
        .SetAttr("padding_after", padding_after, "AttrTypeListInt64")
        .SetAttr("floating_constant_value", float(constant_value), "AttrTypeDouble")
        .SetAttr("integral_constant_value", int(constant_value), "AttrTypeInt64")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )
