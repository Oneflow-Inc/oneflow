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

::tensorflow::Status GrpcRemoteWorker::WorkerConnectDataPlane(
    const WorkerConnectDataPlaneRequest* request,
    WorkerConnectDataPlaneResponse* response) {
  ::grpc::ClientContext ctx;
  return FromGrpcStatus(
      stub_->WorkerConnectDataPlane(&ctx, *request, response));
}

void GrpcRemoteWorker::WorkerConnectDataPlaneAsync(
    const WorkerConnectDataPlaneRequest* request,
    WorkerConnectDataPlaneResponse* response,
    ::tensorflow::StatusCallback done) {
  IssueRequest(request, response, worker_connect_data_plane_, std::move(done));
}

void GrpcRemoteWorker::WorkerInitRuntimeAsync(
    const WorkerInitRuntimeRequest* request,
    WorkerInitRuntimeResponse* response, ::tensorflow::StatusCallback done) {
  IssueRequest(request, response, worker_init_runtime_, std::move(done));
}

void GrpcRemoteWorker::WorkerInitModelAsync(
    const WorkerInitModelRequest* request, WorkerInitModelResponse* response,
    ::tensorflow::StatusCallback done) {
  IssueRequest(request, response, worker_init_model_, std::move(done));
}

void GrpcRemoteWorker::WorkerActivateActorAsync(
    const WorkerActivateActorRequest* request,
    WorkerActivateActorResponse* response, ::tensorflow::StatusCallback done) {
  IssueRequest(request, response, worker_activate_actor_, std::move(done));
}

void GrpcRemoteWorker::WorkerSendRemoteRegstAsync(
    const WorkerSendRemoteRegstRequest* request,
    WorkerSendRemoteRegstResponse* response,
    ::tensorflow::StatusCallback done) {
  IssueRequest(request, response, worker_send_remote_regst_, std::move(done));
}

void GrpcRemoteWorker::WorkerStartActorAsync(
    const WorkerStartActorRequest* request, WorkerStartActorResponse* response,
    ::tensorflow::StatusCallback done) {
  IssueRequest(request, response, worker_start_actor_, std::move(done));
}

::tensorflow::Status GrpcRemoteWorker::WorkerInitDataPlane(
    const WorkerInitDataPlaneRequest* request,
    WorkerInitDataPlaneResponse* response) {
  ::grpc::ClientContext ctx;
  return FromGrpcStatus(stub_->WorkerInitDataPlane(&ctx, *request, response));
}

void GrpcRemoteWorker::WorkerInitDataPlaneAsync(
    const WorkerInitDataPlaneRequest* request,
    WorkerInitDataPlaneResponse* response, ::tensorflow::StatusCallback done) {
  IssueRequest(request, response, worker_init_data_plane_, std::move(done));
}

}  // namespace oneflow
