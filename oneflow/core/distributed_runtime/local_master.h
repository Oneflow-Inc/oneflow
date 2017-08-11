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

#ifndef ONEFLOW_CORE_DISTRIBUTED_RUNTIME_LOCAL_MASTER_H_
#define ONEFLOW_CORE_DISTRIBUTED_RUNTIME_LOCAL_MASTER_H_

#include <memory>

#include "oneflow/core/distributed_runtime/master.h"
#include "oneflow/core/distributed_runtime/master.pb.h"
#include "oneflow/core/distributed_runtime/master_interface.h"

namespace oneflow {

// class Master;

// An implementation of the OneFlow master interface that enables direct
// intraprocess communication between the client and the master implementation.
//
// This master implementation is intended to provide more efficient access to
// a master service that has been created in the same process as the client.
//
// TODO(mrry): Add methods that avoid protobuf encoding the request/response
// objects where this affects performance.
// TODO(mrry): Avoid closure creation/context switch overhead for synchronous
// invocation of Master methods.
// TODO(mrry): Make all potentially blocking Master methods take CallOptions
// for cancellation.
class LocalMaster : public MasterInterface {
 public:
  ~LocalMaster() {}

  ::tensorflow::Status SendJob(const SendJobRequest* request,
                               SendJobResponse* response) override;

 private:
  Master* master_impl_;  // Not owned.

  //// See `LocalMaster::Lookup` for the factory function that creates
  //// objects of this type.
  // LocalMaster(Master* master_impl, const int64 default_timeout_in_ms);

  TF_DISALLOW_COPY_AND_ASSIGN(LocalMaster);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DISTRIBUTED_RUNTIME_LOCAL_MASTER_H_
