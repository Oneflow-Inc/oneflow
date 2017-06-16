#ifndef ONEFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_UTIL_H_
#define ONEFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_UTIL_H_

#include <memory>

#include "grpc++/grpc++.h"
#include "tensorflow/core/lib/core/status.h"

namespace oneflow {

inline ::tensorflow::Status FromGrpcStatus(const ::grpc::Status& s) {
  if (s.ok()) {
    return ::tensorflow::Status::OK();
  } else {
    return ::tensorflow::Status(static_cast<tensorflow::error::Code>(s.error_code()),
                  s.error_message());
  }
}

inline ::grpc::Status ToGrpcStatus(const ::tensorflow::Status& s) {
  if (s.ok()) {
    return ::grpc::Status::OK;
  } else {
    return ::grpc::Status(static_cast<::grpc::StatusCode>(s.code()),
                          s.error_message());
  }
}

typedef std::shared_ptr<::grpc::Channel> SharedGrpcChannelPtr;

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_UTIL_H_
