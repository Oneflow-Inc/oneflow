import oneflow._oneflow_internal


def RangePush(range_name):
    oneflow._oneflow_internal.profiler.RangePush(range_name)


def RangePop():
    oneflow._oneflow_internal.profiler.RangePop()


def ProfilerStart():
    oneflow._oneflow_internal.profiler.ProfilerStart()


def ProfilerStop():
    oneflow._oneflow_internal.profiler.ProfilerStop()
