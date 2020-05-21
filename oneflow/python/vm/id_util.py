from __future__ import absolute_import

import oneflow.python.framework.c_api_util as c_api_util

class IdGenerator(object):
    def NewSymbolId(self):
        raise NotImplementedError 

    def NewObjectId(self):
        raise NotImplementedError 

class PhysicalIdGenerator(IdGenerator):
    def NewSymbolId(self):
        return c_api_util.NewPhysicalSymbolId()

    def NewObjectId(self):
        return c_api_util.NewPhysicalObjectId()

class LogicalIdGenerator(IdGenerator):
    def NewSymbolId(self):
        return c_api_util.NewLogicalSymbolId()

    def NewObjectId(self):
        return c_api_util.NewLogicalObjectId()
