from oneflow.compatible.single_client.framework.distribute import split_sbp

broadcast = oneflow._oneflow_internal.sbp.broadcast()
partial_sum = oneflow._oneflow_internal.sbp.partial_sum()
