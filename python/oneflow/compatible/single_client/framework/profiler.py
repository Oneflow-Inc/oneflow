import oneflow._oneflow_internal

def RangePush(range_name):
    oneflow._oneflow_internal.profiler.RangePush(range_name)

def RangePop():
    oneflow._oneflow_internal.profiler.RangePop()