#include "net/grpc_master_service_impl.h"

#include "grpc++/impl/codegen/async_stream.h"
#include "grpc++/impl/codegen/async_unary_call.h"
#include "grpc++/impl/codegen/channel_interface.h"
#include "grpc++/impl/codegen/client_unary_call.h"
#include "grpc++/impl/codegen/method_handler_impl.h"
#include "grpc++/impl/codegen/rpc_service_method.h"
#include "grpc++/impl/codegen/service_type.h"
#include "grpc++/impl/codegen/sync_stream.h"

namespace oneflow {

namespace grpc {

static const char* grpcMasterService_method_names[] = {
    "/oneflow.MasterService/CreateSession",
    "/oneflow.MasterService/ExtendSession",
    "/oneflow.MasterService/RunStep",
    "/oneflow.MasterService/CloseSession",
    "/oneflow.MasterService/ListDevices",
    "/oneflow.MasterService/Reset",
};

std::unique_ptr<MasterService::Stub> MasterService::NewStub(
    const std::shared_ptr< ::grpc::ChannelInterface>& channel,
    const ::grpc::StubOptions& options) {
  std::unique_ptr<MasterService::Stub> stub(new MasterService::Stub(channel));
  return stub;
}

MasterService::Stub::Stub(
    const std::shared_ptr< ::grpc::ChannelInterface>& channel)
    : channel_(channel),
      rpcmethod_CreateSession_(grpcMasterService_method_names[0],
                               ::grpc::RpcMethod::NORMAL_RPC, channel),
      rpcmethod_ExtendSession_(grpcMasterService_method_names[1],
                               ::grpc::RpcMethod::NORMAL_RPC, channel),
      rpcmethod_RunStep_(grpcMasterService_method_names[2],
                         ::grpc::RpcMethod::NORMAL_RPC, channel),
      rpcmethod_CloseSession_(grpcMasterService_method_names[3],
                              ::grpc::RpcMethod::NORMAL_RPC, channel),
      rpcmethod_ListDevices_(grpcMasterService_method_names[4],
                             ::grpc::RpcMethod::NORMAL_RPC, channel),
      rpcmethod_Reset_(grpcMasterService_method_names[5],
                       ::grpc::RpcMethod::NORMAL_RPC, channel) {}

::grpc::Status MasterService::Stub::CreateSession(
    ::grpc::ClientContext* context, const CreateSessionRequest& request,
    CreateSessionResponse* response) {
  return ::grpc::BlockingUnaryCall(channel_.get(), rpcmethod_CreateSession_,
                                   context, request, response);
}

::grpc::Status MasterService::Stub::ExtendSession(
    ::grpc::ClientContext* context, const ExtendSessionRequest& request,
    ExtendSessionResponse* response) {
  return ::grpc::BlockingUnaryCall(channel_.get(), rpcmethod_ExtendSession_,
                                   context, request, response);
}

::grpc::Status MasterService::Stub::RunStep(::grpc::ClientContext* context,
                                            const RunStepRequest& request,
                                            RunStepResponse* response) {
  return ::grpc::BlockingUnaryCall(channel_.get(), rpcmethod_RunStep_, context,
                                   request, response);
}

::grpc::Status MasterService::Stub::CloseSession(
    ::grpc::ClientContext* context, const CloseSessionRequest& request,
    CloseSessionResponse* response) {
  return ::grpc::BlockingUnaryCall(channel_.get(), rpcmethod_CloseSession_,
                                   context, request, response);
}

::grpc::Status MasterService::Stub::ListDevices(
    ::grpc::ClientContext* context, const ListDevicesRequest& request,
    ListDevicesResponse* response) {
  return ::grpc::BlockingUnaryCall(channel_.get(), rpcmethod_ListDevices_,
                                   context, request, response);
}

::grpc::Status MasterService::Stub::Reset(::grpc::ClientContext* context,
                                          const ResetRequest& request,
                                          ResetResponse* response) {
  return ::grpc::BlockingUnaryCall(channel_.get(), rpcmethod_Reset_, context,
                                   request, response);
}

MasterService::AsyncService::AsyncService() {
  for (int i = 0; i < 6; ++i) {
    AddMethod(new ::grpc::RpcServiceMethod(grpcMasterService_method_names[i],
                                           ::grpc::RpcMethod::NORMAL_RPC,
                                           nullptr));
    ::grpc::Service::MarkMethodAsync(i);
  }
}

MasterService::AsyncService::~AsyncService() {}

}  // namespace grpc

}  // namespace oneflow
