from __future__ import absolute_import

import oneflow.python.framework.inter_user_job as inter_user_job

def LogicalBlob:
    def __init__(self):
        TODO()
        pass
    
    @property
    def op_name(self):
        return self.op_name_

    def pull(self):
        return inter_user_job.pull(self)
