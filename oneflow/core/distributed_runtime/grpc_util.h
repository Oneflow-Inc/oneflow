#ifndef ONEFLOW_CORE_DISTRIBUTED_RUNTIME_GRPC_UTIL_H_
#define ONEFLOW_CORE_DISTRIBUTED_RUNTIME_GRPC_UTIL_H_

#include <memory>
#include "grpc++/grpc++.h"
#include "tensorflow/core/lib/core/status.h"

namespace oneflow {

inline Status FromGrpcStatus(const ::grpc::Status& s) {
  if (s.ok()) {
    return Status::OK();
  } else {
    return Status(static_cast<tensorflow::error::Code>(s.error_code()),
                  s.error_message());
  }
}  // Fromgrpcstatus

inline ::grpc::Status ToGrpcStatus(const ::tensorflow::Status& s) {
  if (s.ok) {
    return ::grpc::Status::OK;
  } else {
    return ::grpc::Status(static_cast<::grpc::StatusCode>(s.code()),
                          s.error_message());
  }
}  // Togrpcstatus

typedef std::shared_ptr<::grpc::Channel> SharedGrpcChannelPtr;

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DISTRIBUTED_RUNTIME_GRPC_UTIL_H_
