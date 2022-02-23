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
#ifndef ONEFLOW_CORE_FRAMEWORK_STREAM_GET_STREAM_ROLE_NAME_H_
#define ONEFLOW_CORE_FRAMEWORK_STREAM_GET_STREAM_ROLE_NAME_H_

#include <glog/logging.h>
#include <string>
#include "oneflow/core/common/stream_role.h"
#include "oneflow/core/common/device_type.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/to_string.h"

namespace oneflow {

struct GetStreamRoleName {
  static Maybe<const char*> Case(StreamRoleCase<StreamRole::kInvalid>) {  // NOLINT
    return "invalid";
  }
  static Maybe<const char*> Case(StreamRoleCase<StreamRole::kCompute>) {
    return "compute";
  }
  static Maybe<const char*> Case(StreamRoleCase<StreamRole::kHost2Device>) {
    return "h2d";
  }
  static Maybe<const char*> Case(StreamRoleCase<StreamRole::kDevice2Host>) {
    return "d2h";
  }
  static Maybe<const char*> Case(StreamRoleCase<StreamRole::kSyncedLaunchedCommNet>) {
    return "synced_launched_comm_net";
  }
  static Maybe<const char*> Case(StreamRoleCase<StreamRole::kAsyncedLaunchedCommNet>) {
    return "asynced_launched_comm_net";
  }
  static Maybe<const char*> Case(StreamRoleCase<StreamRole::kBarrier>) {
    return "barrier";
  }
  static Maybe<const char*> Case(StreamRoleCase<StreamRole::kCriticalSection>) {
    return "critical_section";
  }
  static Maybe<const char*> Case(StreamRoleCase<StreamRole::kLazyJobLauncher>) {
    return "lazy_job_launcher";
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_STREAM_GET_STREAM_ROLE_NAME_H_
