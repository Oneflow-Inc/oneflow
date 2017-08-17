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

#ifndef ONEFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_MASTER_SERVICE_H_
#define ONEFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_MASTER_SERVICE_H_

#include <memory>
#include "grpc++/alarm.h"
#include "grpc++/server_builder.h"

#include "oneflow/core/device/async_cpu_stream.h"
#include "oneflow/core/device/cpu_stream.h"
#include "oneflow/core/distributed_runtime/master.h"
#include "oneflow/core/distributed_runtime/rpc/async_service_interface.h"
#include "oneflow/core/distributed_runtime/rpc/grpc_call.h"
#include "oneflow/core/distributed_runtime/rpc/grpc_master_service_impl.h"
#include "oneflow/core/distributed_runtime/rpc/grpc_util.h"

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/master.pb.h"

namespace grpc {
class ServerBuilder;
}  // namespace grpc

namespace oneflow {

class AsyncServiceInterface;
class Master;

// GrpcMasterService implements the RPC service MasterSerivce.
//
// A GrpcMasterService maintains the state of live graph computation
// sessions, each session orchestrates both local and remote devices
// to carry out the graph computation.
//
// A GrpcMasterService knows ahead of time local devices available as
// client devices.
//
// A GrpcMasterService discovers remote devices in the background and
// keeps track of statistics of those remote devices.
//
// Each session analyzes the graph, places nodes across available
// devices, and ultimately drives the graph computation by initiating
// RunGraph on workers.

class GrpcMasterService : public AsyncServiceInterface {
 public:
  GrpcMasterService(Master* master, ::grpc::ServerBuilder* builder)
      : master_impl_(master), is_shutdown_(false) {
    builder->RegisterService(&master_service_);
    cq_ = builder->AddCompletionQueue();
    cpu_stream_ = new AsyncCpuStream();
  }

  ~GrpcMasterService() override {
    delete shutdown_alarm_;
    delete cpu_stream_;
  }

  void Shutdown() override;

  void HandleRPCsLoop() override;

  void DoWorkLoop() override;

 private:
  Master* master_impl_ = nullptr;  // Not owned.
  std::unique_ptr<::grpc::ServerCompletionQueue> cq_;
  grpc::MasterService::AsyncService master_service_;
  CpuStream* cpu_stream_ = nullptr;

  tensorflow::mutex mu_;
  bool is_shutdown_ GUARDED_BY(mu_);
  ::grpc::Alarm* shutdown_alarm_ = nullptr;

  template<class RequestMessage, class ResponseMessage>
  using MasterCall = Call<GrpcMasterService, grpc::MasterService::AsyncService,
                          RequestMessage, ResponseMessage>;

  void SendJobHandler(MasterCall<SendJobRequest, SendJobResponse>* call);

  void MasterConnectDataPlaneHandler(
      MasterCall<MasterConnectDataPlaneRequest, MasterConnectDataPlaneResponse>*
          call);

  void MasterInitRuntimeHandler(
      MasterCall<MasterInitRuntimeRequest, MasterInitRuntimeResponse>* call);

  void MasterInitModelHandler(
      MasterCall<MasterInitModelRequest, MasterInitModelResponse>* call);

  void MasterActivateActorHandler(
      MasterCall<MasterActivateActorRequest, MasterActivateActorResponse>*
          call);

  void MasterSendRemoteRegstHandler(
      MasterCall<MasterSendRemoteRegstRequest, MasterSendRemoteRegstResponse>*
          call);

  void MasterStartActorHandler(
      MasterCall<MasterStartActorRequest, MasterStartActorResponse>* call);

  void MasterInitDataPlaneHandler(
      MasterCall<MasterInitDataPlaneRequest, MasterInitDataPlaneResponse>*
          call);

  TF_DISALLOW_COPY_AND_ASSIGN(GrpcMasterService);
};
AsyncServiceInterface* NewGrpcMasterService(Master* master,
                                            ::grpc::ServerBuilder* builder);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_MASTER_SERVICE_H_
