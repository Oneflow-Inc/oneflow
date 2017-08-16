#ifndef ONEFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_WORKER_SERVICE_IMPL_H_
#define ONEFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_WORKER_SERVICE_IMPL_H_

#include <grpc++/impl/codegen/async_stream.h>
#include <grpc++/impl/codegen/async_unary_call.h>
#include <grpc++/impl/codegen/proto_utils.h>
#include <grpc++/impl/codegen/rpc_method.h>
#include <grpc++/impl/codegen/service_type.h>
#include <grpc++/impl/codegen/status.h>
#include <grpc++/impl/codegen/stub_options.h>
#include <grpc++/impl/codegen/sync_stream.h>

#include "oneflow/core/distributed_runtime/worker.pb.h"

namespace grpc {

class CompletionQueue;
class Channel;
class RpcService;
class ServerCompletionQueue;
class ServerContext;

}  // namespace grpc

namespace oneflow {

enum class GrpcWorkerMethod {
  kSendPlan,
  kWorkerConnectDataPlane,
  kWorkerInitDataPlane,
};
static const int kGrpcNumWorkerMethods =
    static_cast<int>(GrpcWorkerMethod::kWorkerInitDataPlane) + 1;

const char* GrpcWorkerMethodName(GrpcWorkerMethod id);

namespace grpc {

class WorkerService GRPC_FINAL {
 public:
  class StubInterface {
   public:
    virtual ~StubInterface() {}
    virtual ::grpc::Status SendPlan(::grpc::ClientContext* context,
                                    const SendPlanRequest& request,
                                    SendPlanResponse* response) = 0;
    virtual ::grpc::Status WorkerConnectDataPlane(
        ::grpc::ClientContext* context,
        const WorkerConnectDataPlaneRequest& request,
        WorkerConnectDataPlaneResponse* response) = 0;
    virtual ::grpc::Status WorkerInitDataPlane(
        ::grpc::ClientContext* context,
        const WorkerInitDataPlaneRequest& request,
        WorkerInitDataPlaneResponse* response) = 0;
  };

  class Stub GRPC_FINAL : public StubInterface {
   public:
    Stub(const std::shared_ptr<::grpc::ChannelInterface>& channel);

    ::grpc::Status SendPlan(::grpc::ClientContext* context,
                            const SendPlanRequest& request,
                            SendPlanResponse* response) GRPC_OVERRIDE;

    ::grpc::Status WorkerConnectDataPlane(
        ::grpc::ClientContext* context,
        const WorkerConnectDataPlaneRequest& request,
        WorkerConnectDataPlaneResponse* response) GRPC_OVERRIDE;

    ::grpc::Status WorkerInitDataPlane(
        ::grpc::ClientContext* context,
        const WorkerInitDataPlaneRequest& request,
        WorkerInitDataPlaneResponse* response) GRPC_OVERRIDE;

   private:
    std::shared_ptr<::grpc::ChannelInterface> channel_;
    const ::grpc::RpcMethod rpcmethod_SendPlan_;
    const ::grpc::RpcMethod rpcmethod_WorkerConnectDataPlane_;
    const ::grpc::RpcMethod rpcmethod_WorkerInitDataPlane_;
  };  // class Stub

  static std::unique_ptr<Stub> NewStub(
      const std::shared_ptr<::grpc::ChannelInterface>& channel);

  class AsyncService : public ::grpc::Service {
   public:
    AsyncService();
    virtual ~AsyncService();
    using ::grpc::Service::RequestAsyncUnary;
  };  // Asyncservice
};    // Workerservice
}  // namespace grpc
}  // namespace oneflow
#endif  // ONEFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_WORKER_SERVICE_IMPL_H_
