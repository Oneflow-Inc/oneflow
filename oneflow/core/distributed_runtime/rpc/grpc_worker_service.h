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

#ifndef ONEFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_WORKER_SERVICE_H_
#define ONEFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_WORKER_SERVICE_H_

#include <memory>
#include "grpc++/alarm.h"
#include "grpc++/server_builder.h"

#include "oneflow/core/device/cpu_stream.h"
#include "oneflow/core/device/async_cpu_stream.h"
#include "oneflow/core/distributed_runtime/rpc/async_service_interface.h"
#include "oneflow/core/distributed_runtime/rpc/grpc_call.h"
#include "oneflow/core/distributed_runtime/rpc/grpc_util.h"
#include "oneflow/core/distributed_runtime/rpc/grpc_worker_service_impl.h"
#include "oneflow/core/distributed_runtime/worker.h"

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/worker.pb.h"

namespace grpc {
class ServerBuilder;
}  // namespace grpc

namespace oneflow {

class AsyncServiceInterface;
class Worker;

// GrpcWorkerService implements the RPC service WorkerSerivce.
//
// A GrpcWorkerService maintains the state of live graph computation
// sessions, each session orchestrates both local and remote devices
// to carry out the graph computation.
//
// A GrpcWorkerService knows ahead of time local devices available as
// client devices.
//
// A GrpcWorkerService discovers remote devices in the background and
// keeps track of statistics of those remote devices.
//
// Each session analyzes the graph, places nodes across available
// devices, and ultimately drives the graph computation by initiating
// RunGraph on workers.

class GrpcWorkerService : public AsyncServiceInterface {
 public:
  GrpcWorkerService(Worker* worker, ::grpc::ServerBuilder* builder)
      : worker_impl_(worker), is_shutdown_(false) {
    builder->RegisterService(&worker_service_);
    cq_ = builder->AddCompletionQueue();
    cpu_stream_ = new AsyncCpuStream();
  }

  ~GrpcWorkerService() override {
    delete shutdown_alarm_;
    delete cpu_stream_;
  }

  void Shutdown() override;

  void HandleRPCsLoop() override;

  void DoWorkLoop() override;

 private:
  Worker* worker_impl_ = nullptr;  // Not owned.
  std::unique_ptr<::grpc::ServerCompletionQueue> cq_;
  grpc::WorkerService::AsyncService worker_service_;
  CpuStream *cpu_stream_;

  tensorflow::mutex mu_;
  bool is_shutdown_ GUARDED_BY(mu_);
  ::grpc::Alarm* shutdown_alarm_ = nullptr;

  template<class RequestMessage, class ResponseMessage>
  using WorkerCall = Call<GrpcWorkerService, grpc::WorkerService::AsyncService,
                          RequestMessage, ResponseMessage>;

  void GrpcWorkerService::SendPlanHandler(
      WorkerCall<SendPlanRequest, SendPlanResponse>* call);

  TF_DISALLOW_COPY_AND_ASSIGN(GrpcWorkerService);
};
AsyncServiceInterface* NewGrpcWorkerService(Worker* worker,
                                            ::grpc::ServerBuilder* builder);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_WORKER_SERVICE_H_
