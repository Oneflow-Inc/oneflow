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

#ifndef ONEFLOW_API_COMMON_SCOPE_H_
#define ONEFLOW_API_COMMON_SCOPE_H_

#include <memory>
#include <string>
#include "oneflow/core/common/just.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/session_util.h"
#include "oneflow/core/job/job_conf.cfg.h"
#include "oneflow/core/job/job_conf.pb.h"
#include "oneflow/core/job/scope.h"

namespace oneflow {

inline Maybe<Scope> MakeScope(const JobConfigProto& config_proto, const Device& device) {
  std::shared_ptr<Scope> scope;
  std::shared_ptr<cfg::JobConfigProto> cfg_config_proto =
      std::make_shared<cfg::JobConfigProto>(config_proto);
  JUST(LogicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
    int64_t session_id = 0;
    std::string device_tag = "cpu";
    std::string machine_ids = "0";
    std::string device_ids = "0";
    if (device.type() == "cuda") {
      device_tag = "gpu";
      device_ids = std::to_string(device.device_id());
    }
    scope = JUST(builder->BuildInitialScope(session_id, cfg_config_proto, device_tag,
                                            {machine_ids + ":" + device_ids}, nullptr, false));
    return Maybe<void>::Ok();
  }));
  return scope;
}

}  // namespace oneflow

#endif  // ONEFLOW_API_COMMON_SCOPE_H_
