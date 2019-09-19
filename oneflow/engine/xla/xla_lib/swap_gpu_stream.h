#ifndef TENSORFLOW_COMPILER_JIT_XLA_LIB_SWAP_GPU_STREAM_H_
#define TENSORFLOW_COMPILER_JIT_XLA_LIB_SWAP_GPU_STREAM_H_

#include "tensorflow/stream_executor/stream.h"

namespace se = stream_executor;

namespace xla {

void SwapGpuStreamHandle(se::Stream *stream, void **gpu_stream);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_JIT_XLA_LIB_SWAP_GPU_STREAM_H_
