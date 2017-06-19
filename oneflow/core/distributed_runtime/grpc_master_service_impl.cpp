#include "oneflow/core/distributed_runtime/grpc_master_service_impl.h"

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
  "/oneflow.MasterService/SendGraph",
};

std::unique_ptr<MasterService::Stub> MasterService::NewStub(
    const std::shared_ptr<::grpc::ChannelInterface>& channel,
    const ::grpc::StubOptions& options) {
  std::unique_ptr<MasterService::Stub> stub(new MasterService::Stub(channel));
  return stub;
}

MasterService::Stub::Stub(
    const std::shared_ptr<::grpc::ChannelInterface>& channel)
  : channel_(channel),
    rpcmethod_SendGraph_(grpcMasterService_method_names[0],
                              ::grpc::RpcMethod::NORMAL_RPC, channel) {

}

::grpc::Status MasterService::Stub::SendGraph(
    ::grpc::ClientContext* context, const SendGraphRequest& request,
    SendGraphResponse* response) {
  return ::grpc::BlockingUnaryCall(channel_.get(), rpcmethod_SendGraph_,
                                   context, request, response);
}

MasterService::AsyncService::AsyncService() {
  for (int i = 0; i < 1; ++i) {
    AddMethod(new ::grpc::RpcServiceMethod(grpcMasterService_method_names[i],
                                           ::grpc::RpcMethod::NORMAL_RPC, 
                                           nullptr));
    ::grpc::Service::MarkMethodAsync(i);
  }
}

MasterService::AsyncService::~AsyncService() {}

}  // namespace grpc

}  // namespace oneflow


