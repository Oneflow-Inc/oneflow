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
#ifndef ONEFLOW_CORE_JOB_CLUSTER_OBJECTS_SCOPE_H_
#define ONEFLOW_CORE_JOB_CLUSTER_OBJECTS_SCOPE_H_
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/optional.h"
#include "oneflow/core/job/env_desc.h"
#include "oneflow/core/framework/device.h"

namespace oneflow {

class ParallelDesc;

class EnvGlobalObjectsScope final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EnvGlobalObjectsScope);
  explicit EnvGlobalObjectsScope(const std::string& env_proto_str);
  explicit EnvGlobalObjectsScope(const EnvProto& env_proto);
  ~EnvGlobalObjectsScope();

  Maybe<void> init_is_normal_exit(bool is_normal_exit) {
    CHECK_OR_RETURN(!is_normal_exit_.has_value());
    is_normal_exit_ = is_normal_exit;
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> Init(const EnvProto& env_proto);

 private:
  Optional<bool> is_normal_exit_;
};

Maybe<void> InitRDMA();

Maybe<bool> RDMAIsInitialized();

Maybe<void> DestoryRDMA();

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_CLUSTER_OBJECTS_SCOPE_H_
