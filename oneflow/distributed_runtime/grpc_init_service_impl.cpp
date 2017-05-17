/*
 * grpc_init_service_impl.cpp
 * Copyright (C) 2017 xiaoshu <2012wxs@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#include "distributed_runtime/grpc_init_service_impl.h"

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

static const char* InitService_method_names[] = {
  "/oneflow.InitService/ExchangeMachineInfo",
  "/oneflow.InitService/ExchangeMemoryDesc",
};

std::unique_ptr<InitService::Stub> InitService::NewStub(
    const std::shared_ptr<::grpc::ChannelInterface>& channel,
    const ::grpc::StubOptions& options) {
  std::unique_ptr<InitService::Stub> stub(new InitService::Stub(channel));
  return stub;
}

InitService::Stub::Stub(
    const std::shared_ptr<::grpc::ChannelInterface>& channel)
    : channel_(channel),
      rpcmethod_ExchangeMachineInfo_(InitService_method_names[0], 
                                     ::grpc::RpcMethod::NORMAL_RPC, channel),
      rpcmethod_ExchangeMemoryDesc_(InitService_method_names[1], 
                                      ::grpc::RpcMethod::NORMAL_RPC, channel) {}

::grpc::Status InitService::Stub::ExchangeMachineInfo(
    ::grpc::ClientContext* context, const ::oneflow::ExchangeMachineInfoRequest& request, 
    ::oneflow::ExchangeMachineInfoResponse* response) {
  return ::grpc::BlockingUnaryCall(channel_.get(), rpcmethod_ExchangeMachineInfo_, context, request, response);
}

::grpc::Status InitService::Stub::ExchangeMemoryDesc(
    ::grpc::ClientContext* context, const ::oneflow::ExchangeMemoryDescRequest& request, 
    ::oneflow::ExchangeMemoryDescResponse* response) {
  return ::grpc::BlockingUnaryCall(channel_.get(), rpcmethod_ExchangeMemoryDesc_, context, request, response);
}  

InitService::Service::Service() {
  for (int i = 0; i < 2; ++i) {
    AddMethod(new ::grpc::RpcServiceMethod(InitService_method_names[i],
                                           ::grpc::RpcMethod::NORMAL_RPC,
                                           nullptr));
    ::grpc::Service::MarkMethodAsync(i);
  }
}

}//namespace grpc

}//namespace oneflow



