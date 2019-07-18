#ifndef ONEFLOW_CORE_COMPILER_OF2XLA_XLA_STREAM_H_
#define ONEFLOW_CORE_COMPILER_OF2XLA_XLA_STREAM_H_

#include "oneflow/core/job/resource.pb.h"  // DeviceType
#include "tensorflow/stream_executor/stream.h"

namespace oneflow {
namespace mola {

template <DeviceType device_type>
void SwapStreamHandle(se::Stream *stream, void **cuda_stream);

}  // namespace mola
}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMPILER_OF2XLA_XLA_STREAM_H_
