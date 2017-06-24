#ifndef ONEFLOW_CORE_DISTRIBUTED_RUNTIME_GRPC_CLIENT_CQ_TAG_H_
#define ONEFLOW_CORE_DISTRIBUTED_RUNTIME_GRPC_CLIENT_CQ_TAG_H_

#include "grpc++/grpc++.h"
#include "tensorflow/core/lib/core/status.h"

#include "oneflow/core/distributed_runtime/grpc_util.h"

namespace oneflow {

class GrpcClientCQTag {
 public:
  GrpcClientCQTag() {}
  virtual ~GrpcClientCQTag() {}

  virtual void OnCompleted(bool ok) = 0;
};  // namespace GrpcClientCQTag

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DISTRIBUTED_RUNTIME_GRPC_CLIENT_CQ_TAG_H_
