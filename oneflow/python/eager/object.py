from __future__ import absolute_import

import oneflow.python.eager.symbol as symbol_util
import oneflow

import traceback


class Object(object):
    def __init__(self, object_id, parallel_desc_symbol):
        self.object_id_ = object_id
        self.parallel_desc_symbol_ = parallel_desc_symbol

    @property
    def object_id(self):
        return self.object_id_

    @property
    def parallel_desc_symbol(self):
        return self.parallel_desc_symbol_


class BlobObject(Object):
    def __init__(self, object_id, op_arg_parallel_attr, op_arg_blob_attr, release):
        Object.__init__(self, object_id, op_arg_parallel_attr.parallel_desc_symbol)
        self.op_arg_parallel_attr_ = op_arg_parallel_attr
        self.op_arg_blob_attr_ = op_arg_blob_attr
        self.release_ = []
        if release is not None:
            self.release_.append(release)

    @property
    def op_arg_parallel_attr(self):
        return self.op_arg_parallel_attr_

    @property
    def op_arg_blob_attr(self):
        return self.op_arg_blob_attr_

    def add_releaser(self, release):
        self.release_.append(release)

    def __del__(self):
        for release in self.release_:
            release(self)
        self.release_ = []

    def InitOpArgBlobAttr(self, op_arg_blob_attr):
        assert self.op_arg_blob_attr_ is None
        self.op_arg_blob_attr_ = op_arg_blob_attr
