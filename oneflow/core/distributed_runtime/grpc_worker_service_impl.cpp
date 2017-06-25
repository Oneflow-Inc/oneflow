#include "oneflow/core/distributed_runtime/grpc_worker_service_impl.h"

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
    case GrpcWorkerMethod::kGetStatus:
      return "/oneflow.WorkerService/GetStatus";
    case GrpcWorkerMethod::kGetMachineDesc:
      return "/oneflow.WorkerService/GetMachineDesc";
    case GrpcWorkerMethod::kGetMemoryDesc:
      return "/oneflow.WorkerService/GetMemoryDesc";
    case GrpcWorkerMethod::kSendTaskGraph:
      return "/oneflow.WorkerService/SendTaskGraph";
    case GrpcWorkerMethod::kSendMessage:
      return "/oneflow.WorkerService/SendMessage";
    case GrpcWorkerMethod::kReadData:
      return "/oneflow.WorkerService/ReadData";
  }
}

namespace grpc {

// for client
std::unique_ptr<WorkerService::Stub> WorkerService::NewStub(
    const std::shared_ptr<::grpc::ChannelInterface>& channel) {
  std::unique_ptr<WorkerService::Stub> stub(
      new WorkerService::Stub(channel));
  return stub;
}

WorkerService::Stub::Stub(
    const std::shared_ptr<::grpc::ChannelInterface>& channel)
    : channel_(channel),
      rpcmethod_GetStatus_(GrpcWorkerMethodName(
                             static_cast<GrpcWorkerMethod>(0)),
                             ::grpc::RpcMethod::NORMAL_RPC, channel),
      rpcmethod_GetMachineDesc_(GrpcWorkerMethodName(
                                  static_cast<GrpcWorkerMethod>(1)),
                                  ::grpc::RpcMethod::NORMAL_RPC, channel),
      rpcmethod_GetMemoryDesc_(GrpcWorkerMethodName(
                                static_cast<GrpcWorkerMethod>(2)),
                                ::grpc::RpcMethod::NORMAL_RPC, channel),
      rpcmethod_SendTaskGraph_(GrpcWorkerMethodName(
                                 static_cast<GrpcWorkerMethod>(3)),
                                 ::grpc::RpcMethod::NORMAL_RPC, channel) {}

::grpc::Status WorkerService::Stub::GetStatus(
    ::grpc::ClientContext* context, const GetStatusRequest& request,
    GetStatusResponse* response) {
  return ::grpc::BlockingUnaryCall(channel_.get(), rpcmethod_GetStatus_,
                                   context, request, response);
}

::grpc::Status WorkerService::Stub::GetMachineDesc(
    ::grpc::ClientContext* context, const GetMachineDescRequest& request,
    GetMachineDescResponse* response) {
  return ::grpc::BlockingUnaryCall(channel_.get(), rpcmethod_GetMachineDesc_,
                                    context, request, response);
}

::grpc::Status WorkerService::Stub::GetMemoryDesc(
    ::grpc::ClientContext* context, const GetMemoryDescRequest& request,
    GetMemoryDescResponse* response) {
  return ::grpc::BlockingUnaryCall(channel_.get(), rpcmethod_GetMemoryDesc_,
                                   context, request, response);
}

::grpc::Status WorkerService::Stub::SendTaskGraph(
    ::grpc::ClientContext* context, const SendTaskGraphRequest& request,
    SendTaskGraphResponse* response) {
  return ::grpc::BlockingUnaryCall(channel_.get(),
                                   rpcmethod_SendTaskGraph_,
                                   context, request, response);
}

//for server
WorkerService::AsyncService::AsyncService() {
  for (int i = 0; i < kGrpcNumWorkerMethods; ++i) {
    AddMethod(new ::grpc::RpcServiceMethod(
                GrpcWorkerMethodName(static_cast<GrpcWorkerMethod>(i)),
                ::grpc::RpcMethod::NORMAL_RPC,
                nullptr));
    ::grpc::Service::MarkMethodAsync(i);
  }
}

WorkerService::AsyncService::~AsyncService() {}

}  // namespace grpc

}  // namespace oneflow
