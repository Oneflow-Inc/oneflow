#include "tensorflow/compiler/jit/xla_lib/swap_gpu_stream.h"

#include "tensorflow/stream_executor/gpu/gpu_stream.h"

namespace xla {

void SwapGpuStreamHandle(se::Stream *stream, void **gpu_stream) {
  void **cuda_stream = se::gpu::AsGpuStream(stream)->GpuStreamMemberHack();
  void *tmp_stream = *cuda_stream;
  *cuda_stream = *gpu_stream;
  *gpu_stream = tmp_stream;
}

}  // namespace xla
