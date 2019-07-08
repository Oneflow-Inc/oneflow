from __future__ import absolute_import

import oneflow.python.framework.compile_context as compile_ctx

job_name2input_op_names = {}

job_name2output_op_names = {}

def Init():
    global job_name2input_op_names
    global job_name2output_op_names
    job_name2input_op_names = {
        k, v.op_name for k, v in compile_ctx.job_name2input_logical_blobs.items()
    }
    job_name2output_op_names = {
        k, v.op_name for k, v in compile_ctx.job_name2output_logical_blobs.items()
    }
