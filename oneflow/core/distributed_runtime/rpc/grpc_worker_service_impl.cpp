#include "oneflow/core/distributed_runtime/rpc/grpc_worker_service_impl.h"

#include "grpc++/impl/codegen/async_stream.h"
#include "grpc++/impl/codegen/async_unary_call.h"
#include "grpc++/impl/codegen/channel_interface.h"
#include "grpc++/impl/codegen/client_unary_call.h"
#include "grpc++/impl/codegen/method_handler_impl.h"
#include "grpc++/impl/codegen/rpc_service_method.h"
#include "grpc++/impl/codegen/service_type.h"
#include "grpc++/impl/codegen/sync_stream.h"

namespace oneflow {

const char* GrpcWorkerMethodName(GrpcWorkerMethod id) {
  switch (id) {
    case GrpcWorkerMethod::kSendPlan: return "/oneflow.WorkerService/SendPlan";
  }
}

namespace grpc {

// for client
std::unique_ptr<WorkerService::Stub> WorkerService::NewStub(
    const std::shared_ptr<::grpc::ChannelInterface>& channel) {
  std::unique_ptr<WorkerService::Stub> stub(new WorkerService::Stub(channel));
  return stub;
}

WorkerService::Stub::Stub(
    const std::shared_ptr<::grpc::ChannelInterface>& channel)
    : channel_(channel),
      rpcmethod_SendPlan_(
          GrpcWorkerMethodName(static_cast<GrpcWorkerMethod>(0)),
          ::grpc::RpcMethod::NORMAL_RPC, channel) {}

::grpc::Status WorkerService::Stub::SendPlan(::grpc::ClientContext* context,
                                             const SendPlanRequest& request,
                                             SendPlanResponse* response) {
  return ::grpc::BlockingUnaryCall(channel_.get(), rpcmethod_SendPlan_, context,
                                   request, response);
}

// for server
WorkerService::AsyncService::AsyncService() {
  for (int i = 0; i < kGrpcNumWorkerMethods; ++i) {
    AddMethod(new ::grpc::RpcServiceMethod(
        GrpcWorkerMethodName(static_cast<GrpcWorkerMethod>(i)),
        ::grpc::RpcMethod::NORMAL_RPC, nullptr));
    ::grpc::Service::MarkMethodAsync(i);
  }
}

WorkerService::AsyncService::~AsyncService() {}
}  // namespace grpc
}  // namespace oneflow
