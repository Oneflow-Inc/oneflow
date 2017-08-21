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
    case GrpcWorkerMethod::kWorkerConnectDataPlane:
      return "/oneflow.WorkerService/WorkerConnectDataPlane";
    case GrpcWorkerMethod::kWorkerInitRuntime:
      return "/oneflow.WorkerService/WorkerInitRuntime";
    case GrpcWorkerMethod::kWorkerInitModel:
      return "/oneflow.WorkerService/WorkerInitModel";
    case GrpcWorkerMethod::kWorkerActivateActor:
      return "/oneflow.WorkerService/WorkerActivateActor";
    case GrpcWorkerMethod::kWorkerSendRemoteRegst:
      return "/oneflow.WorkerService/WorkerSendRemoteRegst";
    case GrpcWorkerMethod::kWorkerSendRemoteRegstToConsumer:
      return "/oneflow.WorkerService/WorkerSendRemoteRegstToConsumer";
    case GrpcWorkerMethod::kWorkerStartActor:
      return "/oneflow.WorkerService/WorkerStartActor";
    case GrpcWorkerMethod::kWorkerInitDataPlane:
      return "/oneflow.WorkerService/WorkerInitDataPlane";
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
          ::grpc::RpcMethod::NORMAL_RPC, channel),
      rpcmethod_WorkerConnectDataPlane_(
          GrpcWorkerMethodName(static_cast<GrpcWorkerMethod>(1)),
          ::grpc::RpcMethod::NORMAL_RPC, channel),
      rpcmethod_WorkerInitRuntime_(
          GrpcWorkerMethodName(static_cast<GrpcWorkerMethod>(2)),
          ::grpc::RpcMethod::NORMAL_RPC, channel),
      rpcmethod_WorkerInitModel_(
          GrpcWorkerMethodName(static_cast<GrpcWorkerMethod>(3)),
          ::grpc::RpcMethod::NORMAL_RPC, channel),
      rpcmethod_WorkerActivateActor_(
          GrpcWorkerMethodName(static_cast<GrpcWorkerMethod>(4)),
          ::grpc::RpcMethod::NORMAL_RPC, channel),
      rpcmethod_WorkerSendRemoteRegst_(
          GrpcWorkerMethodName(static_cast<GrpcWorkerMethod>(5)),
          ::grpc::RpcMethod::NORMAL_RPC, channel),
      rpcmethod_WorkerSendRemoteRegstToConsumer_(
          GrpcWorkerMethodName(static_cast<GrpcWorkerMethod>(6)),
          ::grpc::RpcMethod::NORMAL_RPC, channel),
      rpcmethod_WorkerStartActor_(
          GrpcWorkerMethodName(static_cast<GrpcWorkerMethod>(7)),
          ::grpc::RpcMethod::NORMAL_RPC, channel),
      rpcmethod_WorkerInitDataPlane_(
          GrpcWorkerMethodName(static_cast<GrpcWorkerMethod>(8)),
          ::grpc::RpcMethod::NORMAL_RPC, channel) {}

::grpc::Status WorkerService::Stub::SendPlan(::grpc::ClientContext* context,
                                             const SendPlanRequest& request,
                                             SendPlanResponse* response) {
  return ::grpc::BlockingUnaryCall(channel_.get(), rpcmethod_SendPlan_, context,
                                   request, response);
}

::grpc::Status WorkerService::Stub::WorkerConnectDataPlane(
    ::grpc::ClientContext* context,
    const WorkerConnectDataPlaneRequest& request,
    WorkerConnectDataPlaneResponse* response) {
  return ::grpc::BlockingUnaryCall(channel_.get(),
                                   rpcmethod_WorkerConnectDataPlane_, context,
                                   request, response);
}

::grpc::Status WorkerService::Stub::WorkerInitDataPlane(
    ::grpc::ClientContext* context, const WorkerInitDataPlaneRequest& request,
    WorkerInitDataPlaneResponse* response) {
  return ::grpc::BlockingUnaryCall(channel_.get(),
                                   rpcmethod_WorkerInitDataPlane_, context,
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
