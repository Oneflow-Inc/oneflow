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

#include <memory>
#include "grpc++/grpc++.h"

#include "oneflow/core/distributed_runtime/rpc/grpc_remote_worker.h"
#include "tensorflow/core/lib/core/status.h"

namespace oneflow {
::tensorflow::Status GrpcRemoteWorker::SendPlan(const SendPlanRequest* request,
                                                SendPlanResponse* response) {
  ::grpc::ClientContext ctx;
  return FromGrpcStatus(stub_->SendPlan(&ctx, *request, response));
}

void GrpcRemoteWorker::SendPlanAsync(const SendPlanRequest* request,
                                     SendPlanResponse* response,
                                     ::tensorflow::StatusCallback done) {
  IssueRequest(request, response, sendplan_, std::move(done));
}

}  // namespace oneflow
