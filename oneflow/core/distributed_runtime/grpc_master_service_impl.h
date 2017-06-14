#ifndef ONEFLOW_CORE_DISTRIBUTED_RUNTIME_GRPC_MASTER_SERVICE_IMPL_H_
#define ONEFLOW_CORE_DISTRIBUTED_RUNTIME_GRPC_MASTER_SERVICE_IMPL_H_

#include <grpc++/impl/codegen/async_stream.h>
#include <grpc++/impl/codegen/async_unary_call.h>
#include <grpc++/impl/codegen/proto_utils.h>
#include <grpc++/impl/codegen/rpc_method.h>
#include <grpc++/impl/codegen/service_type.h>
#include <grpc++/impl/codegen/status.h>
#include <grpc++/impl/codegen/stub_options.h>
#include <grpc++/impl/codegen/sync_stream.h>

#include <memory>

#include "oneflow/core/distributed_runtime/grpc_serialization_traits.h"
#include "oneflow/core/distributed_runtime/master_service.pb.h"

OF_GRPC_ALLOW_UNLIMITED_MESSAGE_SIZE(oneflow::SendGraphRequest);

namespace grpc {
class CompletionQueue;
class Channel;
class RpcService;
class ServerCompletionQueue;
class ServerContext;
}

namespace oneflow {

enum class GrpcMasterMethod {
  kSendGraph
};

static const int kGrpcNumMasterMethods =
  static_cast<int> (GrpcMasterMethod::kSendGraph) + 1;

const char* GrpcMasterMethodName(GrpcMasterMethod id);

namespace grpc {

class MasterService GRPC_FINAL {
 public:
  class StubInterface {
   public:
    virtual ~StubInterface() {}
    virtual ::grpc::Status SendGraphSync(::grpc::ClientContext* context,
                                     const SendGraphRequest& request,
                                     SendGraphResponse* response) = 0;
  };  // Stubinterface

  class Stub GRPC_FINAL : public StubInterface {
   public:
    Stub(const std::shared_ptr<::grpc::ChannelInterface>& channel);
    ::grpc::Status SendGraphSync(::grpc::ClientContext* context,
                                 const SendGraphRequest& request,
                                 SendGraphResponse* response) GRPC_OVERRIDE;
   private:
    std::shared_ptr<::grpc::ChannelInterface> channel_;
    const ::grpc::RpcMethod rpcmethod_SendGraph_;
  };  // Stub

  static std::unique_ptr<Stub> NewStub(
      const std::shared_ptr<::grpc::ChannelInterface>& channel,
      const ::grpc::StubOptions& options = ::grpc::StubOptions());

  class AsyncService : public ::grpc::Service {
   public:
    AsyncService();
    virtual ~AsyncService();
    using ::grpc::Service::RequestAsyncUnary;
  };  // Asyncservice
};  // Masterservice

}  // namespace grpc

}  // namespace oneflow


#endif  // ONEFLOW_CORE_DISTRIBUTED_RUNTIME_GRPC_MASTER_SERVICE_IMPL_H_
