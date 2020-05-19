#include "glog/logging.h"

#include "oneflow/xrt/platform.h"

#ifdef WITH_CUDA
#include "cuda_runtime.h"
#endif

namespace oneflow {
namespace xrt {

namespace platform {

int GetDeviceId(const XrtDevice &device) {
  switch (device) {
    case XrtDevice::CPU_X86: return 0;
    case XrtDevice::GPU_CUDA: {
#ifdef WITH_CUDA
      int device_id = 0;
      CHECK_EQ(cudaSuccess, cudaGetDevice(&device_id));
      return device_id;
#endif
    }
    case XrtDevice::GPU_CL:
    // TODO(hjchen2)
    case XrtDevice::CPU_ARM:
    // TODO(hjchen2)
    case XrtDevice::GPU_ARM:
      // TODO(hjchen2)
      return 0;
  }
  return 0;  // Compiler warning free
}

}  // namespace platform

}  // namespace xrt
}  // namespace oneflow
