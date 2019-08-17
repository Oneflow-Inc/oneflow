from __future__ import absolute_import

from oneflow.core.register.logical_blob_id_pb2 import LogicalBlobId
from oneflow.python.framework.remote_blob import RemoteBlob

class Blob(RemoteBlob):
    def __init__(self, dl_net, logical_blob_name):
        lbi = LogicalBlobId()
        lbi.op_name = logical_blob_name.split('/')[0]
        lbi.blob_name = logical_blob_name.split('/')[1]
        RemoteBlob.__init__(self, lbi)
        self.dl_net_ = dl_net
        self.logical_blob_name_ = logical_blob_name

    # getters
    def dl_net(self):
        return self.dl_net_

    @property
    def logical_blob_name(self):
        return self.logical_blob_name_
    
    def __str__(self):
        return self.logical_blob_name()
