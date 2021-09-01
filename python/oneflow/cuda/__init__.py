import oneflow as flow


def is_available() -> bool:
    r"""Returns a bool indicating if CUDA is currently available."""
    if not hasattr(flow._oneflow_internal, 'CudaGetDeviceCount'):
        return False
    # This function never throws and returns 0 if driver is missing or can't
    # be initialized
    return flow._oneflow_internal.CudaGetDeviceCount() > 0