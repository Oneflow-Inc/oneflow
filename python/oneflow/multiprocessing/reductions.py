from multiprocessing.reduction import ForkingPickler

import oneflow as flow
from oneflow.nn.parameter import Parameter
from oneflow.framework.tensor import Tensor


try:
    # Early load resource_sharer to prevent a partially initialized instance
    # from being inherited in a forked child process. The reduce_storage method
    # requires this module indirectly through DupFd(). The built-in mp.Queue
    # class pickles arguments in a background thread which may overlap with the
    # fork.
    import multiprocessing.resource_sharer
except ImportError:
    pass


def reduce_tensor(tensor):
    print("\nreductions.py >>>>>>>>>>>>>>>>> reduce_tensor")
    return flow.Tensor(tensor.numpy())

def reduce_local_tensor(tensor):
    print("\nreductions.py >>>>>>>>>>>>>>>>> reduce_local_tensor")
    return tensor

def init_reductions():
    ForkingPickler.register(Tensor, reduce_tensor)
    ForkingPickler.register(flow._oneflow_internal.Tensor, reduce_local_tensor)
    ForkingPickler.register(Parameter, reduce_tensor)
    ForkingPickler.register(flow._oneflow_internal.nn.Parameter, reduce_tensor)
