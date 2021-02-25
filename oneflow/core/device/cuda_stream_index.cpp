#include "oneflow/core/device/cuda_stream_index.h"

namespace oneflow {

REGISTER_STREAM_INDEX_GENERATOR(DeviceType::kGPU, CudaStreamIndexGenerator);

}  // namespace oneflow
