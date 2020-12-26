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
#include "oneflow/core/job/job_conf.cfg.h"
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

int64_t JobDesc::piece_num_of_experiment_phase() const {
  return job_conf_.exp_run_conf().piece_num_of_experiment_phase();
}

bool JobDesc::enable_experiment_run() const {
  return job_conf_.exp_run_conf().enable_experiment_run();
}

int64_t JobDesc::TotalBatchNum() const { return job_conf_.total_batch_num(); }
int64_t JobDesc::NumOfPiecesInBatch() const { return 1; }

JobDesc::JobDesc(const JobConfigProto& job_conf, int64_t job_id)
    : job_conf_(job_conf), job_id_(job_id), symbol_id_(Error::SymbolIdUninitialized()) {
  CHECK_JUST(Init());
}

Maybe<JobDesc> JobDesc::New(int64_t symbol_id, const JobConfigProto& job_conf) {
  auto job_desc = std::make_shared<JobDesc>(job_conf);
  job_desc->symbol_id_ = Maybe<int64_t>(symbol_id);
  return job_desc;
}

Maybe<void> JobDesc::Init() {
  cfg_job_conf_.reset(new cfg::JobConfigProto(job_conf_));
#ifndef WITH_RDMA
  CHECK_NOTNULL_OR_RETURN((Global<ResourceDesc, ForSession>::Get()));
  CHECK_EQ_OR_RETURN((Global<ResourceDesc, ForSession>::Get()->use_rdma()), false)
      << "Please compile ONEFLOW with RDMA";
#endif
  int64_t piece_exp = job_conf_.exp_run_conf().piece_num_of_experiment_phase();
  if (job_conf_.has_train_conf()) {
    if (piece_exp == -1) { piece_exp = 19 * NumOfPiecesInBatch(); }
    piece_exp = std::max(piece_exp, NumOfPiecesInBatch());
    piece_exp = std::min(piece_exp, job_conf_.total_batch_num() * NumOfPiecesInBatch());
  } else {
    if (piece_exp == -1) { piece_exp = 19; }
  }
  LOG(INFO) << "Set piece_num_of_experiment_phase " << piece_exp;
  job_conf_.mutable_exp_run_conf()->set_piece_num_of_experiment_phase(piece_exp);
#ifndef WITH_CUDA
  CHECK_EQ_OR_RETURN((Global<ResourceDesc, ForSession>::Get()->GpuDeviceNum()), 0);
#endif
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
  Global<JobDesc>::New(job_conf, job_id);
}

GlobalJobDescScope::~GlobalJobDescScope() { Global<JobDesc>::Delete(); }

const JobDesc& GlobalJobDesc() { return *Global<JobDesc>::Get(); }

bool IsPullJob(const std::string& job_name, const InterUserJobInfo& inter_user_job_info) {
  for (const auto& pair : inter_user_job_info.output_or_var_op_name2pull_job_name()) {
    if (pair.second == job_name) { return true; }
  }
  return false;
}

bool IsPushJob(const std::string& job_name, const InterUserJobInfo& inter_user_job_info) {
  for (const auto& pair : inter_user_job_info.input_or_var_op_name2push_job_name()) {
    if (pair.second == job_name) { return true; }
  }
  if (job_name == inter_user_job_info.global_model_init_job_name()) { return true; }
  if (job_name == inter_user_job_info.global_model_load_job_name()) { return true; }
  if (job_name == inter_user_job_info.global_model_save_job_name()) { return true; }
  return false;
}

}  // namespace oneflow
