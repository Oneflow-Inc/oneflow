from __future__ import absolute_import

from oneflow.python.framework.oneflow import compile_only
from oneflow.python.framework.oneflow import run
from oneflow.python.framework.decorator_util import remote
from oneflow.python.framework.decorator_util import static_assert
from oneflow.python.framework.decorator_util import main
from oneflow.python.framework.val import val
from oneflow.python.framework.var import var
from oneflow.python.framework.inter_user_job import pull

from oneflow.python.framework import config
from oneflow.python.framework.config import compose_config
## frequently used config api
from oneflow.python.framework.config import machine
from oneflow.python.framework.config import ctrl_port
from oneflow.python.framework.config import data_port
from oneflow.python.framework.config import gpu_device_num
from oneflow.python.framework.config import cpu_device_num
from oneflow.python.framework.config import comm_net_worker_num
from oneflow.python.framework.config import max_mdsave_worker_num
from oneflow.python.framework.config import use_rdma
from oneflow.python.framework.config import rdma_mem_block_mbyte
from oneflow.python.framework.config import rdma_recv_msg_buf_mbyte
from oneflow.python.framework.config import model_load_snapshot_path
from oneflow.python.framework.config import model_save_snapshots_path
from oneflow.python.framework.config import config_train_by_func
from oneflow.python.framework.config import batch_size
from oneflow.python.framework.config import default_data_type
from oneflow.python.framework.config import data_part_num
from oneflow.python.framework.config import enable_cudnn
from oneflow.python.framework.config import cudnn_buf_limit_mbyte
from oneflow.python.framework.config import enable_mem_sharing
from oneflow.python.framework.config import enable_inplace
from oneflow.python.framework.config import enable_nccl
from oneflow.python.framework.config import use_nccl_inter_node_communication
from oneflow.python.framework.config import all_reduce_group_num
from oneflow.python.framework.config import all_reduce_lazy_ratio
from oneflow.python.framework.config import all_reduce_group_min_mbyte
from oneflow.python.framework.config import all_reduce_group_size_warmup
from oneflow.python.framework.config import all_reduce_fp16
from oneflow.python.framework.config import concurrency_width

import oneflow.python.framework.dtype as dtype
for x in dir(dtype):
    if x.startswith('_') == False: locals()[x] = getattr(dtype, x)
del x

del absolute_import
del python
del core
del dtype
