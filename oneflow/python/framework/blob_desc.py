from __future__ import absolute_import

import oneflow.core.common.data_type_pb2 as data_type_util
import oneflow.python.framework.distribute as distribute_util
import oneflow.python.lib.core.traceinfo as traceinfo
import copy
import traceback

class BlobDesc(object):
    def __init__(self, lbi,
                 distribute = distribute_util.auto(),
                 disable_boxing = None):
        self.lbi_ = lbi
        self.lbn_ = lbi.op_name + "/" + lbi.blob_name
        self.distribute_ = distribute
        self.disable_boxing_ = disable_boxing
        self.stack_ = traceinfo.GetStackInfoExcludeOneflowPythonFile()
        self.location_ = "".join(traceback.format_list(self.stack_))

    @property
    def location(self): return self.location_

    @property
    def lbi(self): return self.lbi_

    @property
    def op_name(self): return self.lbi_.op_name

    @property
    def blob_name(self): return self.lbi_.blob_name

    @property
    def shape(self): return self.static_shape

    @property
    def static_shape(self):
        raise NotImplementedError

    @property
    def dtype(self):
        raise NotImplementedError

    @property
    def batch_axis(self):
        raise NotImplementedError

    def has_batch_axis(self):
        batch_axis = self.batch_axis
        ret = batch_axis is not None
        if ret: assert type(batch_axis) is int
        return ret

    @property
    def is_dynamic(self):
        raise NotImplementedError

    @property
    def disable_boxing(self):
        return self.disable_boxing_

    @property
    def is_tensor_list(self):
        raise NotImplementedError
    
    @property
    def parallel_conf(self):
        raise NotImplementedError

    def with_boxing_disabled(self, val = True):
        ret = self.Clone()
        ret.disable_boxing_ = val
        return ret

    def with_distribute(self, distribute):
        ret = self.Clone()
        ret.distribute_ = distribute
        return ret

    def with_split_distribute(self, axis):
        return self.with_distribute(distribute_util.split(axis))

    def with_broadcast_distribute(self):
        return self.with_distribute(distribute_util.broadcast())

    @property
    def distribute(self):
        distribute_util.assert_is_valid_distribute(self.distribute_)
        return self.distribute_

    @property
    def logical_blob_name(self):
        return self.lbn_ + self._Distribute2Str() + self._DisableBoxing2Str()

    def Clone(self):
        return copy.deepcopy(self)

    def _Distribute2Str(self):
        if type(self.distribute_) is distribute_util.AutoDistribute:
            return ""
        elif type(self.distribute_) is distribute_util.SplitDistribute:
            return ":S" + str(self.distribute_.axis)
        elif type(self.distribute_) is distribute_util.BroadcastDistribute:
            return ":B"
        else:
            raise NotImplementedError

    def _DisableBoxing2Str(self):
        if self.disable_boxing_ is None: return ""
        if self.disable_boxing_ is False: return "|0"
        if self.disable_boxing_ is True: return "|1"
        raise NotImplementedError
