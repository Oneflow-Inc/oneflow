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
#ifndef ONEFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_REMOTE_MASTER_H_
#define ONEFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_REMOTE_MASTER_H_

#include <memory>
#include "grpc++/grpc++.h"
#include "oneflow/core/distributed_runtime/master.pb.h"
#include "oneflow/core/distributed_runtime/master_interface.h"
#include "oneflow/core/distributed_runtime/rpc/grpc_master_service_impl.h"
#include "oneflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/lib/core/status.h"

namespace oneflow {

class GrpcRemoteMaster : public MasterInterface {
 public:
  explicit GrpcRemoteMaster(
      const std::shared_ptr<::grpc::Channel>& client_channel)
      : stub_(grpc::MasterService::NewStub(client_channel)) {}

  ~GrpcRemoteMaster() {}

  ::tensorflow::Status SendJob(const SendJobRequest* request,
                               SendJobResponse* response) override;

  ::tensorflow::Status MasterConnectDataPlane(
      const MasterConnectDataPlaneRequest* request,
      MasterConnectDataPlaneResponse* response) override;

  ::tensorflow::Status MasterInitRuntime(
      const MasterInitRuntimeRequest* request,
      MasterInitRuntimeResponse* response) override;

  ::tensorflow::Status MasterInitModel(
      const MasterInitModelRequest* request,
      MasterInitModelResponse* response) override;

  ::tensorflow::Status MasterActivateActor(
      const MasterActivateActorRequest* request,
      MasterActivateActorResponse* response) override;

  ::tensorflow::Status MasterSendRemoteRegst(
      const MasterSendRemoteRegstRequest* request,
      MasterSendRemoteRegstResponse* response) override;

  ::tensorflow::Status MasterStartActor(
      const MasterStartActorRequest* request,
      MasterStartActorResponse* response) override;

  ::tensorflow::Status MasterInitDataPlane(
      const MasterInitDataPlaneRequest* request,
      MasterInitDataPlaneResponse* response) override;

 private:
  std::unique_ptr<grpc::MasterService::Stub> stub_;
};  // GrpcRemoteMaster

}  // namespace oneflow
#endif  // ONEFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_REMOTE_MASTER_H_
