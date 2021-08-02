import tvm
from tvm import relay

xrt_backends = ["xla", "trt", "tvm"]

def set_xrt(config, xrts=[]):
    for xrt in xrts:
        if xrt == "xla":
            config.use_xla_jit(True)
        if xrt == "trt":
            config.use_tensorrt(True)
        if xrt == "tvm":
            config.use_tvm(True)