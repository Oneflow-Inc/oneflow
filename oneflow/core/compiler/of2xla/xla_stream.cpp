#include "tensorflow/compiler/jit/xla_lib/swap_gpu_stream.h"
#include "oneflow/core/compiler/of2xla/xla_stream.h"

namespace oneflow {
namespace mola {

template <DeviceType device_type>
void SwapStreamHandle(se::Stream *stream, void **cuda_stream) {
  LOG(FATAL) << "Should not call this unimplemented function.";
}

template <>
void SwapStreamHandle<DeviceType::kCPU>(se::Stream *stream,
                                        void **cuda_stream) {
  // Do nothing
}

template <>
void SwapStreamHandle<DeviceType::kGPU>(se::Stream *stream,
                                        void **cuda_stream) {
  xla::SwapGpuStreamHandle(stream, cuda_stream);
}

}  // namespace mola
}  // namespace oneflow
