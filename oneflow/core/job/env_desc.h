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
#ifndef ONEFLOW_CORE_JOB_CLUSTER_DESC_H_
#define ONEFLOW_CORE_JOB_CLUSTER_DESC_H_

#include "oneflow/core/job/env.pb.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

class EnvDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EnvDesc);
  explicit EnvDesc(const EnvProto& env_proto) : env_proto_(env_proto) {}
  ~EnvDesc() = default;

  const EnvProto& env_proto() const { return env_proto_; }
  const Machine& machine(int32_t idx) const { return env_proto_.machine(idx); }
  int32_t ctrl_port() const { return env_proto_.ctrl_port(); }
  int32_t data_port() const { return env_proto_.data_port(); }
  size_t TotalMachineNum() const;
  int64_t GetMachineId(const std::string& addr) const;

 private:
  EnvProto env_proto_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_CLUSTER_DESC_H_
