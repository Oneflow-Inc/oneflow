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
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/job_set.pb.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/persistence/hadoop/hadoop_file_system.h"
#include "oneflow/core/graph/graph.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/job/job_builder.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/global_for.h"

namespace oneflow {

namespace {

void CheckFunctionConfig(const JobConfigProto& job_conf) {
  const auto& attr_name2attr_def = GlobalFunctionConfigDef().attr_name2attr_def();
  for (const auto& pair : job_conf.flag_name2flag_value()) {
    const auto& iter = attr_name2attr_def.find(pair.first);
    CHECK(iter != attr_name2attr_def.end());
    CHECK_EQ(iter->second.default_val().value_case(), pair.second.value_case());
  }
}

}  // namespace

JobDesc::JobDesc(const JobConfigProto& job_conf, int64_t job_id)
    : job_conf_(job_conf), job_id_(job_id), symbol_id_(NullOpt) {
  CHECK_JUST(Init());
  Singleton<ResourceDesc, ForSession>::Get()->DumpCudnnConf(job_conf);
}

Maybe<JobDesc> JobDesc::New(int64_t symbol_id, const JobConfigProto& job_conf) {
  auto job_desc = std::make_shared<JobDesc>(job_conf);
  job_desc->symbol_id_ = symbol_id;
  return job_desc;
}

Maybe<void> JobDesc::Init() {
  CheckFunctionConfig(job_conf_);
  return Maybe<void>::Ok();
}

const AttrValue& JobDesc::GetFunctionFlagVal(const std::string& field_name) const {
  const auto& iter = job_conf_.flag_name2flag_value().find(field_name);
  if (iter != job_conf_.flag_name2flag_value().end()) { return iter->second; }
  const auto& attr_name2attr_def = GlobalFunctionConfigDef().attr_name2attr_def();
  const auto& def_iter = attr_name2attr_def.find(field_name);
  CHECK(def_iter != attr_name2attr_def.end());
  return def_iter->second.default_val();
}

bool IsInterfaceOpConf(const OperatorConf& op_conf) {
  return IsClassRegistered<int32_t, IsInterfaceOpConf4OpTypeCase>(op_conf.op_type_case());
}

GlobalJobDescScope::GlobalJobDescScope(const JobConfigProto& job_conf, int64_t job_id) {
  if (Singleton<JobDesc>::Get() != nullptr) { Singleton<JobDesc>::Delete(); }
  Singleton<JobDesc>::New(job_conf, job_id);
}

GlobalJobDescScope::~GlobalJobDescScope() { Singleton<JobDesc>::Delete(); }

const JobDesc& GlobalJobDesc() { return *Singleton<JobDesc>::Get(); }

}  // namespace oneflow
