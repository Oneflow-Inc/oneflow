#ifndef ONEFLOW_CORE_DISTRIBUTED_RUNTIME_GRPC_REMOTE_MASTER_H_
#define ONEFLOW_CORE_DISTRIBUTED_RUNTIME_GRPC_REMOTE_MASTER_H_

#include <memory>

#include "grpc++/grpc++.h"

#include "oneflow/core/distributed_runtime/grpc_master_service_impl.h"
#include "oneflow/core/distributed_runtime/master_service.pb.h"
#include "oneflow/core/distributed_runtime/grpc_util.h"
#include "tensorflow/core/lib/core/status.h"

namespace oneflow {

class GrpcRemoteMaster {
 public:
  explicit GrpcRemoteMaster(std::shared_ptr<::grpc::Channel> client_channel)
      : stub_(grpc::MasterService::NewStub(client_channel)) {}

  ~GrpcRemoteMaster() {}

  tensorflow::Status SendGraphSync(const SendGraphRequest* request,
                       SendGraphResponse* response) {
    ::grpc::ClientContext ctx;
    ctx.set_fail_fast(false);
    return FromGrpcStatus(stub_->SendGraphSync(&ctx, *request, response));
  }

 private:
  std::unique_ptr<grpc::MasterService::Stub> stub_;
  ::grpc::Status status;
};  // Grpcremotemaster

}  // namespace oneflow


#endif  // ONEFLOW_CORE_DISTRIBUTED_RUNTIME_GRPC_REMOTE_MASTER_H_
