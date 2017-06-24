#ifndef ONEFLOW_CORE_DISTRIBUTED_RUNTIME_GRPC_INIT_SERVICE_IMPL_H_
#define ONEFLOW_CORE_DISTRIBUTED_RUNTIME_GRPC_INIT_SERVICE_IMPL_H_

#include <grpc++/impl/codegen/async_stream.h>
#include <grpc++/impl/codegen/async_unary_call.h>
#include <grpc++/impl/codegen/proto_utils.h>
#include <grpc++/impl/codegen/rpc_method.h>
#include <grpc++/impl/codegen/service_type.h>
#include <grpc++/impl/codegen/status.h>
#include <grpc++/impl/codegen/stub_options.h>
#include <grpc++/impl/codegen/sync_stream.h>

//#include "oneflow/core/distributed_runtime/tensor_coding.h"
//#include "oneflow/core/distributed_runtime/grpc_serialization_traits.h"
#include "oneflow/core/distributed_runtime/worker_service.pb.h"

//OF_GRPC_ALLOW_UNLIMITED_MESSAGE_SIZE(oneflow::ReadDataResponse);

namespace grpc {

class CompletionQueue;
class Channel;
class RpcService;
class ServerCompletionQueue;
class ServerContext;

}  // namespace grpc

namespace oneflow {

enum class GrpcWorkerMethod {
  kGetStatus,
  kGetMachineDesc,
  kGetMemoryDesc,
  kSendTaskGraph,
  kSendMessage,
  kReadData,
};
static const int kGrpcNumWorkerMethods =
  static_cast<int>(GrpcWorkerMethod::kReadData) + 1;

const char* GrpcWorkerMethodName(GrpcWorkerMethod id);

namespace grpc{

class WorkerService GRPC_FINAL {
 public:
  class StubInterface {
   public:
    virtual ~StubInterface() {}
    virtual ::grpc::Status GetStatus(::grpc::ClientContext* context,
                             const GetStatusRequest& request,
                             GetStatusResponse* response) = 0;
    virtual ::grpc::Status GetMachineDesc(::grpc::ClientContext* context,
                                  const GetMachineDescRequest& request,
                                  GetMachineDescResponse* response) = 0;
    virtual ::grpc::Status GetMemoryDesc(::grpc::ClientContext* context,
                                 const GetMemoryDescRequest& request,
                                 GetMemoryDescResponse* response) = 0;
    virtual ::grpc::Status SendTaskGraph(::grpc::ClientContext* context,
                                 const SendTaskGraphRequest& request,
                                 SendTaskGraphResponse* response) = 0;
  };

  class Stub GRPC_FINAL : public StubInterface {
   public:
    Stub(const std::shared_ptr<::grpc::ChannelInterface>& channel);

    ::grpc::Status GetStatus(::grpc::ClientContext* context,
                             const GetStatusRequest& request,
                             GetStatusResponse* response) GRPC_OVERRIDE;
    ::grpc::Status GetMachineDesc(::grpc::ClientContext* context,
                                  const GetMachineDescRequest& request,
                                  GetMachineDescResponse* response) GRPC_OVERRIDE;
    ::grpc::Status GetMemoryDesc(::grpc::ClientContext* context,
                                 const GetMemoryDescRequest& request,
                                 GetMemoryDescResponse* response) GRPC_OVERRIDE;
    ::grpc::Status SendTaskGraph(::grpc::ClientContext* context,
                                 const SendTaskGraphRequest& request,
                                 SendTaskGraphResponse* response) GRPC_OVERRIDE;

   private:
    std::shared_ptr<::grpc::ChannelInterface> channel_;
    const ::grpc::RpcMethod rpcmethod_GetStatus_;
    const ::grpc::RpcMethod rpcmethod_GetMachineDesc_;
    const ::grpc::RpcMethod rpcmethod_GetMemoryDesc_;
    const ::grpc::RpcMethod rpcmethod_SendTaskGraph_;
  };  // class Stub

  static std::unique_ptr<Stub> NewStub(
    const std::shared_ptr<::grpc::ChannelInterface>& channel);

  class AsyncService : public ::grpc::Service {
   public:
    AsyncService();
    virtual ~AsyncService();

    using ::grpc::Service::RequestAsyncUnary;
  };  // Asyncservice
};  // Workerservice

}  // namespace grpc

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DISTRIBUTED_RUNTIME_GRPC_INIT_SERVICE_IMPL_H_
