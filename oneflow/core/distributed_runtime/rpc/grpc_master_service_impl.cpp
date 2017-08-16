/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "oneflow/core/distributed_runtime/rpc/grpc_master_service_impl.h"

#include "grpc++/impl/codegen/async_stream.h"
#include "grpc++/impl/codegen/async_unary_call.h"
#include "grpc++/impl/codegen/channel_interface.h"
#include "grpc++/impl/codegen/client_unary_call.h"
#include "grpc++/impl/codegen/method_handler_impl.h"
#include "grpc++/impl/codegen/rpc_service_method.h"
#include "grpc++/impl/codegen/service_type.h"
#include "grpc++/impl/codegen/sync_stream.h"

namespace oneflow {

const char* GrpcMasterMethodName(GrpcMasterMethod id) {
  switch (id) {
    case GrpcMasterMethod::kSendJob: return "/oneflow.MasterService/SendJob";
    case GrpcMasterMethod::kMasterConnectDataPlane:
      return "/oneflow.MasterService/MasterConnectDataPlane";
    case GrpcMasterMethod::kMasterInitDataPlane:
      return "/oneflow.MasterService/MasterInitDataPlane";
  }
}  // GrpcMasterMethodName

namespace grpc {

std::unique_ptr<MasterService::Stub> MasterService::NewStub(
    const std::shared_ptr<::grpc::ChannelInterface>& channel) {
  std::unique_ptr<MasterService::Stub> stub(new MasterService::Stub(channel));
  return stub;
}

MasterService::Stub::Stub(
    const std::shared_ptr<::grpc::ChannelInterface>& channel)
    : channel_(channel),
      rpcmethod_SendJob_(GrpcMasterMethodName(static_cast<GrpcMasterMethod>(0)),
                         ::grpc::RpcMethod::NORMAL_RPC, channel),
      rpcmethod_MasterConnectDataPlane_(
          GrpcMasterMethodName(static_cast<GrpcMasterMethod>(1)),
          ::grpc::RpcMethod::NORMAL_RPC, channel),
      rpcmethod_MasterInitDataPlane_(
          GrpcMasterMethodName(static_cast<GrpcMasterMethod>(2)),
          ::grpc::RpcMethod::NORMAL_RPC, channel) {}

::grpc::Status MasterService::Stub::SendJob(::grpc::ClientContext* context,
                                            const SendJobRequest& request,
                                            SendJobResponse* response) {
  return ::grpc::BlockingUnaryCall(channel_.get(), rpcmethod_SendJob_, context,
                                   request, response);
}

::grpc::Status MasterService::Stub::MasterConnectDataPlane(
    ::grpc::ClientContext* context,
    const MasterConnectDataPlaneRequest& request,
    MasterConnectDataPlaneResponse* response) {
  return ::grpc::BlockingUnaryCall(channel_.get(),
                                   rpcmethod_MasterConnectDataPlane_, context,
                                   request, response);
}
::grpc::Status MasterService::Stub::MasterInitDataPlane(
    ::grpc::ClientContext* context, const MasterInitDataPlaneRequest& request,
    MasterInitDataPlaneResponse* response) {
  return ::grpc::BlockingUnaryCall(channel_.get(),
                                   rpcmethod_MasterInitDataPlane_, context,
                                   request, response);
}

MasterService::AsyncService::AsyncService() {
  for (int i = 0; i < kGrpcNumMasterMethods; ++i) {
    AddMethod(new ::grpc::RpcServiceMethod(
        GrpcMasterMethodName(static_cast<GrpcMasterMethod>(i)),
        ::grpc::RpcMethod::NORMAL_RPC, nullptr));
    ::grpc::Service::MarkMethodAsync(i);
  }
}

MasterService::AsyncService::~AsyncService() {}
}  // namespace grpc
}  // namespace oneflow
