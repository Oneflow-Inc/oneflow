from oneflow.framework.distribute import split_sbp as split

broadcast = oneflow._oneflow_internal.sbp.broadcast()
partial_sum = oneflow._oneflow_internal.sbp.partial_sum()
