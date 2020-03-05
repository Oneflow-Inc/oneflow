from __future__ import absolute_import

import oneflow as flow
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.id_util as id_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util

from oneflow.python.oneflow_export import oneflow_export


@oneflow_export('categorical_ordinal_encode')
def categorical_ordinal_encode(table, size, x, hash_precomputed=True, name=None):
    assert hash_precomputed is True
    return (
        flow.user_op_builder(name or id_util.UniqueStr("CategoricalOrdinalEncode_"))
            .Op("CategoricalOrdinalEncode")
            .Input("in", [x])
            .Input("table", [table])
            .Input("size", [size])
            .Output("out")
            .SetAttr("hash_precomputed", hash_precomputed, "AttrTypeBool")
            .Build()
            .RemoteBlobList()[0]
    )


@oneflow_export('layers.categorical_ordinal_encoder')
def categorical_ordinal_encoder(x, capacity, hash_precomputed=True, name=None):
    assert hash_precomputed is True
    name_prefix = name if name is not None else id_util.UniqueStr('CategoricalOrdinalEncoder_')
    dtype = x.dtype
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
    return categorical_ordinal_encode(table=table, size=size, x=x,
                                      name='{}-Lookup'.format(name_prefix))
