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
#ifndef ONEFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_REMOTE_WORKER_H_
#define ONEFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_REMOTE_WORKER_H_

#include <memory>
#include "grpc++/grpc++.h"
#include "oneflow/core/distributed_runtime/rpc/grpc_util.h"
#include "oneflow/core/distributed_runtime/rpc/grpc_worker_service_impl.h"
#include "oneflow/core/distributed_runtime/worker.pb.h"
#include "oneflow/core/distributed_runtime/worker_interface.h"
#include "tensorflow/core/lib/core/status.h"

namespace oneflow {

class GrpcRemoteWorker : public WorkerInterface {
 public:
  explicit GrpcRemoteWorker(
      const std::shared_ptr<::grpc::Channel>& client_channel)
      : stub_(grpc::WorkerService::NewStub(client_channel)) {}

  ~GrpcRemoteWorker() {}

  ::tensorflow::Status SendPlan(const SendPlanRequest* request,
                                SendPlanResponse* response) override;

 private:
  std::unique_ptr<grpc::WorkerService::Stub> stub_;
};  // GrpcRemoteWorker

}  // namespace oneflow
#endif  // ONEFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_REMOTE_WORKER_H_
