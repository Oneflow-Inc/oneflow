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

def ClearObject4BlobName(blob_name):
    assert HasObject4BlobName(blob_name)
    global blob_name2object
    del blob_name2object[blob_name]

blob_name2object = {}
