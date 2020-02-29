from __future__ import absolute_import

from functools import reduce
import operator

import oneflow as flow
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.distribute as distribute_util
import oneflow.python.framework.id_util as id_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util

from oneflow.python.oneflow_export import oneflow_export


@oneflow_export('experimental.categorical_hash_table_lookup')
def categorical_hash_table_lookup(keys, values, size, x, hash_precomputed=True, name=None):
    assert hash_precomputed is True
    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, 'name', name if name is not None else id_util.UniqueStr('CategoricalHashTableLookup_'))
    setattr(op_conf.categorical_hash_table_lookup_conf, 'key', keys.logical_blob_name)
    setattr(op_conf.categorical_hash_table_lookup_conf, 'value', values.logical_blob_name)
    setattr(op_conf.categorical_hash_table_lookup_conf, 'size', size.logical_blob_name)
    setattr(op_conf.categorical_hash_table_lookup_conf, 'in', x.logical_blob_name)
    setattr(op_conf.categorical_hash_table_lookup_conf, 'out', 'out')
    compile_context.CurJobAddOp(op_conf)
    out_lbi = logical_blob_id_util.LogicalBlobId()
    setattr(out_lbi, "op_name", op_conf.name)
    setattr(out_lbi, "blob_name", 'out')
    return remote_blob_util.RemoteBlob(out_lbi)


@oneflow_export('experimental.layers.categorical_hash_table')
def categorical_hash_table(x, capacity, value_dtype=None, hash_precomputed=True, name=None):
    assert hash_precomputed is True
    name_prefix = name if name is not None else id_util.UniqueStr('CategoricalHashTable_')
    key_dtype = x.dtype
    value_dtype = value_dtype or key_dtype
    table_key = flow.get_variable(
        name="{}-Key".format(name_prefix),
        shape=(capacity,),
        dtype=key_dtype,
        initializer=flow.constant_initializer(0, dtype=key_dtype),
        trainable=False,
    )
    table_value = flow.get_variable(
        name="{}-Value".format(name_prefix),
        shape=(capacity,),
        dtype=value_dtype,
        initializer=flow.constant_initializer(0, dtype=key_dtype),
        trainable=False,
    )
    table_size = flow.get_variable(
        name="{}-Size".format(name_prefix),
        shape=(1,),
        dtype=value_dtype,
        initializer=flow.constant_initializer(0, dtype=key_dtype),
        trainable=False,
    )
    return categorical_hash_table_lookup(keys=table_key, values=table_value, size=table_size, x=x,
                                         name='{}-Lookup'.format(name_prefix))
