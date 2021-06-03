/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_CORE_CONTROL_BOOTSTRAP_CLIENT_H_
#define ONEFLOW_CORE_CONTROL_BOOTSTRAP_CLIENT_H_

#include "oneflow/core/control/rpc_client.h"
#include "oneflow/core/job/env_desc.h"

namespace oneflow {

class BootstrapClient : public RpcClient {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BootstrapClient);
  virtual ~BootstrapClient() override = default;

 protected:
  friend class Global<BootstrapClient>;
  BootstrapClient() = default;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_CONTROL_BOOTSTRAP_CLIENT_H_
