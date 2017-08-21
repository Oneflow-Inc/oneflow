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

#ifndef ONEFLOW_CORE_DISTRIBUTED_RUNTIME_WORKER_INTERFACE_H_
#define ONEFLOW_CORE_DISTRIBUTED_RUNTIME_WORKER_INTERFACE_H_

#include "oneflow/core/distributed_runtime/worker.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace oneflow {

// Abstract interface for communicating with the OneFlow Worker service.
//
// This interface supports both RPC-based master implementations, and
// in-process master implementations that do not require an RPC
// roundtrip.
class WorkerInterface {
 public:
  virtual ~WorkerInterface() {}
  virtual ::tensorflow::Status SendPlan(const SendPlanRequest* request,
                                        SendPlanResponse* response) = 0;
  virtual void SendPlanAsync(const SendPlanRequest* request,
                             SendPlanResponse* response,
                             ::tensorflow::StatusCallback done) = 0;

  virtual ::tensorflow::Status WorkerConnectDataPlane(
      const WorkerConnectDataPlaneRequest* request,
      WorkerConnectDataPlaneResponse* response) = 0;

  virtual void WorkerConnectDataPlaneAsync(
      const WorkerConnectDataPlaneRequest* request,
      WorkerConnectDataPlaneResponse* response,
      ::tensorflow::StatusCallback done) = 0;

  virtual void WorkerInitRuntimeAsync(const WorkerInitRuntimeRequest* request,
                                      WorkerInitRuntimeResponse* response,
                                      ::tensorflow::StatusCallback done) = 0;

  virtual void WorkerInitModelAsync(const WorkerInitModelRequest* request,
                                    WorkerInitModelResponse* response,
                                    ::tensorflow::StatusCallback done) = 0;

  virtual void WorkerActivateActorAsync(
      const WorkerActivateActorRequest* request,
      WorkerActivateActorResponse* response,
      ::tensorflow::StatusCallback done) = 0;

  virtual void WorkerSendRemoteRegstAsync(
      const WorkerSendRemoteRegstRequest* request,
      WorkerSendRemoteRegstResponse* response,
      ::tensorflow::StatusCallback done) = 0;

  virtual void WorkerSendRemoteRegstToConsumerAsync(
      const WorkerSendRemoteRegstToConsumerRequest* request,
      WorkerSendRemoteRegstToConsumerResponse* response,
      ::tensorflow::StatusCallback done) = 0;

  virtual void WorkerStartActorAsync(const WorkerStartActorRequest* request,
                                     WorkerStartActorResponse* response,
                                     ::tensorflow::StatusCallback done) = 0;

  virtual ::tensorflow::Status WorkerInitDataPlane(
      const WorkerInitDataPlaneRequest* request,
      WorkerInitDataPlaneResponse* response) = 0;

  virtual void WorkerInitDataPlaneAsync(
      const WorkerInitDataPlaneRequest* request,
      WorkerInitDataPlaneResponse* response,
      ::tensorflow::StatusCallback done) = 0;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DISTRIBUTED_RUNTIME_WORKER_INTERFACE_H_
