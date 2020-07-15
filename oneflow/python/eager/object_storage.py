from __future__ import absolute_import


def HasSharedOpKernelObject4ParallelConfSymbol(parallel_conf_sym):
    global parallel_conf_symbol2shared_opkernel_object
    return id(parallel_conf_sym) in parallel_conf_symbol2shared_opkernel_object


def GetSharedOpKernelObject4ParallelConfSymbol(parallel_conf_sym):
    global parallel_conf_symbol2shared_opkernel_object
    return parallel_conf_symbol2shared_opkernel_object[id(parallel_conf_sym)]


def SetSharedOpKernelObject4ParallelConfSymbol(
    parallel_conf_sym, shared_opkernel_object
):
    assert not HasSharedOpKernelObject4ParallelConfSymbol(parallel_conf_sym)
    global parallel_conf_symbol2shared_opkernel_object
    parallel_conf_symbol2shared_opkernel_object[
        id(parallel_conf_sym)
    ] = shared_opkernel_object


parallel_conf_symbol2shared_opkernel_object = {}
