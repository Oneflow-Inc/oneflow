from __future__ import absolute_import

import oneflow as flow
import oneflow.python.framework.id_util as id_util
from oneflow.python.oneflow_export import oneflow_export


@oneflow_export('categorical_ordinal_encode')
def categorical_ordinal_encode(table, size, input_tensor, hash_precomputed=True, name=None):
    assert hash_precomputed is True
    return (
        flow.user_op_builder(name or id_util.UniqueStr("CategoricalOrdinalEncode_"))
            .Op("CategoricalOrdinalEncode")
            .Input("in", [input_tensor])
            .Input("table", [table])
            .Input("size", [size])
            .Output("out")
            .SetAttr("hash_precomputed", hash_precomputed, "AttrTypeBool")
            .Build()
            .InferAndTryRun()
            .RemoteBlobList()[0]
    )


@oneflow_export('layers.categorical_ordinal_encoder')
def categorical_ordinal_encoder(input_tensor, capacity, hash_precomputed=True, name=None):
    assert hash_precomputed is True
    name_prefix = name if name is not None else id_util.UniqueStr('CategoricalOrdinalEncoder_')
    dtype = input_tensor.dtype
    table = flow.get_variable(
        name="{}-Table".format(name_prefix),
        shape=(capacity * 2,),
        dtype=dtype,
        initializer=flow.constant_initializer(0, dtype=dtype),
        trainable=False,
    )
    size = flow.get_variable(
        name="{}-Size".format(name_prefix),
        shape=(1,),
        dtype=dtype,
        initializer=flow.constant_initializer(0, dtype=dtype),
        trainable=False,
    )
    return categorical_ordinal_encode(table=table, size=size, input_tensor=input_tensor,
                                      name='{}-Encode'.format(name_prefix))
