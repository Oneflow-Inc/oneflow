#ifndef ONEFLOW_RUNTIME_DISTRIBUTED_RUNTIME_GRPC_TENSOR_CODING_H_
#define ONEFLOW_RUNTIME_DISTRIBUTED_RUNTIME_GRPC_TENSOR_CODING_H_

#include "oneflow/core/distributed_runtime/tensor.h"

namespace grpc {
class ByteBuffer;
}  // namespace grpc

namespace oneflow {
class ReadDataResponse;

namespace grpc {
// encoding RecvTensorResponse format data
void EncodeRecvTensorResponseToByteBuffer(const ReadDataResponse& proto,
                                          ::grpc::ByteBuffer* result);

void EncodeTensorToByteBuffer(bool is_dead, const Tensor& val, ::grpc::ByteBuffer* result);

}  // namespace grpc

}  // namespace oneflow

#endif  // ONEFLOW_RUNTIME_DISTRIBUTED_RUNTIME_GRPC_TENSOR_CODING_H_
