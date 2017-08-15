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
#ifndef ONEFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_MASTER_SERVICE_IMPL_H_
#define ONEFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_MASTER_SERVICE_IMPL_H_

#include <grpc++/impl/codegen/async_stream.h>
#include <grpc++/impl/codegen/async_unary_call.h>
#include <grpc++/impl/codegen/proto_utils.h>
#include <grpc++/impl/codegen/rpc_method.h>
#include <grpc++/impl/codegen/service_type.h>
#include <grpc++/impl/codegen/status.h>
#include <grpc++/impl/codegen/stub_options.h>
#include <grpc++/impl/codegen/sync_stream.h>

#include <memory>
#include "oneflow/core/distributed_runtime/master.pb.h"

namespace grpc {
class CompletionQueue;
class Channel;
class RpcService;
class ServerCompletionQueue;
class ServerContext;
}  // namespace grpc

namespace oneflow {

enum class GrpcMasterMethod {
  kSendJob,
};

static const int32_t kGrpcNumMasterMethods =
    static_cast<int32_t>(GrpcMasterMethod::kSendJob) + 1;

const char* GrpcMasterMethodName(GrpcMasterMethod id);

namespace grpc {

class MasterService GRPC_FINAL {
 public:
  class StubInterface {
   public:
    virtual ~StubInterface() {}
    virtual ::grpc::Status SendJob(::grpc::ClientContext* context,
                                   const SendJobRequest& request,
                                   SendJobResponse* response) = 0;
  };  // Stubinterface
  class Stub GRPC_FINAL : public StubInterface {
   public:
    Stub(const std::shared_ptr<::grpc::ChannelInterface>& channel);
    ::grpc::Status SendJob(::grpc::ClientContext* context,
                           const SendJobRequest& request,
                           SendJobResponse* response) GRPC_OVERRIDE;

   private:
    std::shared_ptr<::grpc::ChannelInterface> channel_;
    const ::grpc::RpcMethod rpcmethod_SendJob_;
  };  // Stub

  static std::unique_ptr<Stub> NewStub(
      const std::shared_ptr<::grpc::ChannelInterface>& channel);

  class AsyncService : public ::grpc::Service {
   public:
    AsyncService();
    virtual ~AsyncService();
    using ::grpc::Service::RequestAsyncUnary;
  };  // Asyncservice
};    // Masterservice
}  // namespace grpc
}  // namespace oneflow

#endif  // ONEFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_MASTER_SERVICE_IMPL_H_
