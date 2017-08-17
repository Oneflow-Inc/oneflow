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

#include "oneflow/core/distributed_runtime/rpc/grpc_remote_master.h"
#include "tensorflow/core/lib/core/status.h"

namespace oneflow {
::tensorflow::Status GrpcRemoteMaster::SendJob(const SendJobRequest* request,
                                               SendJobResponse* response) {
  ::grpc::ClientContext ctx;
  return FromGrpcStatus(stub_->SendJob(&ctx, *request, response));
}

::tensorflow::Status GrpcRemoteMaster::MasterConnectDataPlane(
    const MasterConnectDataPlaneRequest* request,
    MasterConnectDataPlaneResponse* response) {
  ::grpc::ClientContext ctx;
  return FromGrpcStatus(
      stub_->MasterConnectDataPlane(&ctx, *request, response));
}

::tensorflow::Status GrpcRemoteMaster::MasterInitRuntime(
    const MasterInitRuntimeRequest* request,
    MasterInitRuntimeResponse* response) {
  ::grpc::ClientContext ctx;
  return FromGrpcStatus(stub_->MasterInitRuntime(&ctx, *request, response));
}

::tensorflow::Status GrpcRemoteMaster::MasterInitModel(
    const MasterInitModelRequest* request, MasterInitModelResponse* response) {
  ::grpc::ClientContext ctx;
  return FromGrpcStatus(stub_->MasterInitModel(&ctx, *request, response));
}

::tensorflow::Status GrpcRemoteMaster::MasterActivateActor(
    const MasterActivateActorRequest* request,
    MasterActivateActorResponse* response) {
  ::grpc::ClientContext ctx;
  return FromGrpcStatus(stub_->MasterActivateActor(&ctx, *request, response));
}

::tensorflow::Status GrpcRemoteMaster::MasterSendRemoteRegst(
    const MasterSendRemoteRegstRequest* request,
    MasterSendRemoteRegstResponse* response) {
  ::grpc::ClientContext ctx;
  return FromGrpcStatus(stub_->MasterSendRemoteRegst(&ctx, *request, response));
}

::tensorflow::Status GrpcRemoteMaster::MasterStartActor(
    const MasterStartActorRequest* request,
    MasterStartActorResponse* response) {
  ::grpc::ClientContext ctx;
  return FromGrpcStatus(stub_->MasterStartActor(&ctx, *request, response));
}

::tensorflow::Status GrpcRemoteMaster::MasterInitDataPlane(
    const MasterInitDataPlaneRequest* request,
    MasterInitDataPlaneResponse* response) {
  ::grpc::ClientContext ctx;
  return FromGrpcStatus(stub_->MasterInitDataPlane(&ctx, *request, response));
}

}  // namespace oneflow
