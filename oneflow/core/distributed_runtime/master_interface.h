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

#ifndef ONEFLOW_CORE_DISTRIBUTED_RUNTIME_MASTER_INTERFACE_H_
#define ONEFLOW_CORE_DISTRIBUTED_RUNTIME_MASTER_INTERFACE_H_

#include "oneflow/core/distributed_runtime/master.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace oneflow {

// Abstract interface for communicating with the OneFlow Master service.
//
// This interface supports both RPC-based master implementations, and
// in-process master implementations that do not require an RPC
// roundtrip.
class MasterInterface {
 public:
  virtual ~MasterInterface() {}
  virtual ::tensorflow::Status SendJob(const SendJobRequest* request,
                                       SendJobResponse* response) = 0;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DISTRIBUTED_RUNTIME_MASTER_INTERFACE_H_
