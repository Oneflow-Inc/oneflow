from __future__ import absolute_import

import oneflow
from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.framework.id_util as id_util

@oneflow_export("pad")
def pad(x, paddings, constant_value=0, name=None):
    paddings_list = []
    if isinstance(paddings, (list, tuple)):
        assert len(paddings) == len(x.static_shape), ValueError(
            "paddings must be the same size of input dims"
        )
        for p in paddings:
            assert isinstance(p, (list, tuple)) and len(p) == 2, ValueError(
                "the elem of paddings must be a tuple or a list with length of 2"
            )
            paddings_list.append(p[0])
            paddings_list.append(p[1])
    else:
        raise ValueError("paddings must be a tuple or a list.")
    return (
        oneflow.user_op_builder(name if name is not None else id_util.UniqueStr("Pad_"))
        .Op("pad")
        .Input("x", [x])
        .Output("y")
        .SetAttr("paddings", paddings_list, "AttrTypeListInt64")
        .SetAttr("constant_value", float(constant_value), "AttrTypeFloat")
        .Build()
        .RemoteBlobList()[0]
    )


@oneflow_export("pad_grad")
def pad_grad(x, paddings, constant_value=0, name=None):
    paddings_list = []
    if isinstance(paddings, (list, tuple)):
        assert len(paddings) == len(x.static_shape), ValueError(
            "paddings must be the same size of input dims"
        )
        for p in paddings:
            assert isinstance(p, (list, tuple)) and len(p) == 2, ValueError(
                "the elem of paddings must be a tuple or a list with length of 2"
            )
            paddings_list.append(p[0])
            paddings_list.append(p[1])
    else:
        raise ValueError("paddings must be a tuple or a list.")
    return (
        oneflow.user_op_builder(name if name is not None else id_util.UniqueStr("PadGrad_"))
        .Op("pad_grad")
        .Input("dy", [x])
        .Output("dx")
        .SetAttr("paddings", paddings_list, "AttrTypeListInt64")
        .SetAttr("constant_value", float(constant_value), "AttrTypeFloat")
        .Build()
        .RemoteBlobList()[0]
    )
