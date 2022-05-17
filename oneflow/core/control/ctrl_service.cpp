/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/control/ctrl_service.h"

namespace oneflow {

namespace {

template<size_t method_index>
const grpc::internal::RpcMethod BuildOneRpcMethod(std::shared_ptr<grpc::ChannelInterface> channel) {
  return grpc::internal::RpcMethod(GetMethodName(static_cast<CtrlMethod>(method_index)),
                                   grpc::internal::RpcMethod::NORMAL_RPC, channel);
}

template<size_t... method_indices>
std::array<const grpc::internal::RpcMethod, kCtrlMethodNum> BuildRpcMethods(
    std::index_sequence<method_indices...>, std::shared_ptr<grpc::ChannelInterface> channel) {
  return {BuildOneRpcMethod<method_indices>(channel)...};
}

constexpr int64_t kDefaultGrpcMaxMessageByteSize = -1;

}  // namespace

CtrlService::Stub::Stub(std::shared_ptr<grpc::ChannelInterface> channel)
    : rpcmethods_(BuildRpcMethods(std::make_index_sequence<kCtrlMethodNum>{}, channel)),
      channel_(channel) {}

std::unique_ptr<CtrlService::Stub> CtrlService::NewStub(const std::string& addr) {
  grpc::ChannelArguments ch_args;
  int64_t max_msg_byte_size =
      ParseIntegerFromEnv("ONEFLOW_GRPC_MAX_MESSAGE_BYTE_SIZE", kDefaultGrpcMaxMessageByteSize);
  ch_args.SetInt(GRPC_ARG_MAX_MESSAGE_LENGTH, max_msg_byte_size);
  return std::make_unique<Stub>(
      grpc::CreateCustomChannel(addr, grpc::InsecureChannelCredentials(), ch_args));
}

CtrlService::AsyncService::AsyncService() {
  for (int32_t i = 0; i < kCtrlMethodNum; ++i) {
    AddMethod(new grpc::internal::RpcServiceMethod(GetMethodName(static_cast<CtrlMethod>(i)),
                                                   grpc::internal::RpcMethod::NORMAL_RPC, nullptr));
    grpc::Service::MarkMethodAsync(i);
  }
}

}  // namespace oneflow
