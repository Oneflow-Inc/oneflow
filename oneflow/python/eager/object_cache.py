from contextlib import contextmanager

def HasSharedOpKernelObject4ParallelConfSymbol(parallel_conf_sym):
    global parallel_conf_symbol2shared_opkernel_object
    return id(parallel_conf_sym) in parallel_conf_symbol2shared_opkernel_object

def GetSharedOpKernelObject4ParallelConfSymbol(parallel_conf_sym):
    global parallel_conf_symbol2shared_opkernel_object
    return parallel_conf_symbol2shared_opkernel_object[id(parallel_conf_sym)]

def SetSharedOpKernelObject4ParallelConfSymbol(parallel_conf_sym, shared_opkernel_object):
    assert not HasSharedOpKernelObject4ParallelConfSymbol(parallel_conf_sym)
    global parallel_conf_symbol2shared_opkernel_object
    parallel_conf_symbol2shared_opkernel_object[id(parallel_conf_sym)] = shared_opkernel_object

parallel_conf_symbol2shared_opkernel_object = {}

def HasObject4BlobName(blob_name):
    global blob_name2object
    return blob_name in blob_name2object

def GetObject4BlobName(blob_name):
    global blob_name2object
    assert HasObject4BlobName(blob_name), "blob_name %s not found" % blob_name
    return blob_name2object[blob_name]

def SetObject4BlobName(blob_name, obj):
    assert not HasObject4BlobName(blob_name), blob_name
    global blob_name2object
    blob_name2object[blob_name] = obj

@contextmanager
def BnInOp2BlobObjectScope(op_attribute):
    bn_in_op2blob_object = {}
    for ibn in op_attribute.input_bns:
        lbi = op_attribute.bn_in_op2lbi[ibn]
        bn_in_op2blob_object[ibn] = GetObject4BlobName("%s/%s"%(lbi.op_name, lbi.blob_name))
    yield bn_in_op2blob_object
    for obn in op_attribute.output_bns:
        lbi = op_attribute.bn_in_op2lbi[obn]
        SetObject4BlobName("%s/%s"%(lbi.op_name, lbi.blob_name), bn_in_op2blob_object[obn])

def ClearObject4BlobName(blob_name):
    assert HasObject4BlobName(blob_name)
    global blob_name2object
    del blob_name2object[blob_name]

blob_name2object = {}
