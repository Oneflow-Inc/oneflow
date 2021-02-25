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
#include "oneflow/core/job/model_io_v2_job.h"
#include "oneflow/core/operator/interface_op_util.h"
#include "oneflow/core/common/buffer_manager.h"

namespace oneflow {

namespace {

bool CompareVariableOpConf(const VariableOpConf& lhs, const VariableOpConf& rhs) {
  if (lhs.has_random_seed() && rhs.has_random_seed()) {
    CHECK_EQ(lhs.random_seed(), rhs.random_seed());
  }
  VariableOpConf var_conf_a(lhs);
  VariableOpConf var_conf_b(rhs);
  var_conf_a.clear_tick();
  var_conf_b.clear_tick();
  var_conf_a.clear_out();
  var_conf_b.clear_out();
  return PbMd::Equals(var_conf_a, var_conf_b);
}

OperatorConf GenForeignInputOpConf(const std::string& job_name, const int64_t input_size) {
  OperatorConf foreign_input_op_conf{};
  foreign_input_op_conf.set_name("System-Push-ForeignInput_" + NewUniqueId());
  ForeignInputOpConf* foreign_input_conf = foreign_input_op_conf.mutable_foreign_input_conf();
  foreign_input_conf->set_out("out");
  foreign_input_conf->set_ofblob_buffer_name(GetForeignInputBufferName(job_name));
  InterfaceBlobConf* blob_conf = foreign_input_conf->mutable_blob_conf();
  *blob_conf->mutable_shape()->mutable_dim()->Add() = input_size;
  blob_conf->set_is_dynamic(true);
  blob_conf->set_data_type(DataType::kInt8);
  blob_conf->mutable_split_axis()->clear_value();
  return foreign_input_op_conf;
}

void SetModelIoDefaultJobConf(JobConfigProto* job_conf, const std::string& job_name) {
  job_conf->set_job_name(job_name);
  job_conf->mutable_predict_conf();
  job_conf->set_total_batch_num(1);
}

OperatorConf GenTickOpConf(const std::string& op_name) {
  OperatorConf tick_op_conf{};
  tick_op_conf.set_name(op_name);
  tick_op_conf.mutable_tick_conf()->set_out("out");
  return tick_op_conf;
}

OperatorConf CloneVariableOpConf(const OperatorConf& variable_op_conf) {
  OperatorConf new_var_op_conf(variable_op_conf);
  new_var_op_conf.mutable_variable_conf()->clear_tick();
  return new_var_op_conf;
}

std::string GetVariableLbn(const OperatorConf& variable_op_conf) {
  return GenLogicalBlobName(variable_op_conf.name(), variable_op_conf.variable_conf().out());
}

void FilterVariableOps(const std::vector<std::shared_ptr<Job>>& jobs,
                       HashMap<std::string, OperatorConf>* var_op_name2op_conf) {
  FOR_RANGE(int64_t, job_id, 0, jobs.size()) {
    for (const OperatorConf& op_conf : jobs.at(job_id)->net().op()) {
      if (op_conf.has_variable_conf()) {
        if (var_op_name2op_conf->find(op_conf.name()) == var_op_name2op_conf->end()) {
          CHECK(var_op_name2op_conf->emplace(op_conf.name(), op_conf).second);
        } else {
          CHECK(CompareVariableOpConf(var_op_name2op_conf->at(op_conf.name()).variable_conf(),
                                      op_conf.variable_conf()));
        }
      }
    }
  }
}

void MakeModelInitJob(
    const std::string& job_name, Job* job,
    const HashMap<std::string, OperatorConf>& var_op_name2op_conf,
    const HashMap<std::string, ParallelBlobConf>& var_op_name2parallel_blob_conf) {
  auto* flag_name2flag_value = job->mutable_job_conf()->mutable_flag_name2flag_value();
  (*flag_name2flag_value)["__is_user_function__"].set_at_bool(false);
  SetModelIoDefaultJobConf(job->mutable_job_conf(), job_name);
  Global<InterUserJobInfo>::Get()->set_global_model_init_job_name(job_name);
  JobBuilder job_builder(job);
  const ParallelConf master_parallel_conf = GenParallelConfOfCpuZeroOnMaster();
  const OperatorConf tick_op_conf = GenTickOpConf("System-ModelInit-Tick");
  const OperatorConf foreign_input_op_conf = GenForeignInputOpConf(job_name, 1);
  job_builder.AddOps(master_parallel_conf, {foreign_input_op_conf, tick_op_conf});
  if (var_op_name2op_conf.empty()) { return; }
  std::string prev_post_model_init_tick_lbn = GenLogicalBlobName(
      foreign_input_op_conf.name(), foreign_input_op_conf.foreign_input_conf().out());
  for (const auto& pair : var_op_name2op_conf) {
    const auto& var_op_name = pair.first;
    const OperatorConf& variable_op_conf = pair.second;
    OperatorConf new_var_op_conf = CloneVariableOpConf(variable_op_conf);
    const ParallelBlobConf& parallel_blob_conf = var_op_name2parallel_blob_conf.at(var_op_name);
    OperatorConf model_init_op_conf{};
    model_init_op_conf.set_name("System-ModelInit-" + var_op_name);
    ModelInitV2OpConf* model_init_conf = model_init_op_conf.mutable_model_init_v2_conf();
    model_init_conf->set_ref(GetVariableLbn(variable_op_conf));
    *model_init_conf->mutable_original_variable_conf() = variable_op_conf.variable_conf();
    model_init_conf->set_variable_op_name(variable_op_conf.name());
    model_init_conf->set_tick(prev_post_model_init_tick_lbn);
    *model_init_conf->mutable_out() = "out";
    prev_post_model_init_tick_lbn =
        GenLogicalBlobName(model_init_op_conf.name(), model_init_conf->out());
    job_builder.AddOps(parallel_blob_conf.parallel_conf(), {new_var_op_conf, model_init_op_conf});
  }
}

void MakeModelLoadJob(
    const std::string& job_name, Job* job,
    const HashMap<std::string, OperatorConf>& var_op_name2op_conf,
    const HashMap<std::string, ParallelBlobConf>& var_op_name2parallel_blob_conf) {
  auto* flag_name2flag_value = job->mutable_job_conf()->mutable_flag_name2flag_value();
  (*flag_name2flag_value)["__is_user_function__"].set_at_bool(false);
  SetModelIoDefaultJobConf(job->mutable_job_conf(), job_name);
  Global<InterUserJobInfo>::Get()->set_global_model_load_job_name(job_name);
  JobBuilder job_builder(job);
  const ParallelConf master_parallel_conf = GenParallelConfOfCpuZeroOnMaster();
  const OperatorConf tick_op_conf = GenTickOpConf("System-ModelLoad-Tick");
  const OperatorConf foreign_input_op_conf = GenForeignInputOpConf(job_name, 65536);
  job_builder.AddOps(master_parallel_conf, {foreign_input_op_conf, tick_op_conf});
  if (var_op_name2op_conf.empty()) { return; }
  std::string prev_post_model_load_tick_lbn = GenLogicalBlobName(
      foreign_input_op_conf.name(), foreign_input_op_conf.foreign_input_conf().out());
  for (const auto& pair : var_op_name2op_conf) {
    const auto& var_op_name = pair.first;
    const OperatorConf& variable_op_conf = pair.second;
    const ParallelConf& variable_op_parallel_conf =
        var_op_name2parallel_blob_conf.at(var_op_name).parallel_conf();
    const VariableOpConf& origin_variable_conf = variable_op_conf.variable_conf();
    CHECK_NE(origin_variable_conf.data_type(), DataType::kInvalidDataType);
    OperatorConf new_var_op_conf = CloneVariableOpConf(variable_op_conf);
    OperatorConf model_load_op_conf{};
    model_load_op_conf.set_name("System-ModelLoad-" + var_op_name);
    ModelLoadV2OpConf* model_load_conf = model_load_op_conf.mutable_model_load_v2_conf();
    model_load_conf->set_path(GenLogicalBlobName(foreign_input_op_conf.name(),
                                                 foreign_input_op_conf.foreign_input_conf().out()));
    model_load_conf->set_ref(GetVariableLbn(new_var_op_conf));
    *model_load_conf->mutable_variable_op_name() = var_op_name;
    *model_load_conf->mutable_original_variable_conf() = origin_variable_conf;
    *model_load_conf->mutable_out() = "out";
    *model_load_conf->mutable_tick() = prev_post_model_load_tick_lbn;
    prev_post_model_load_tick_lbn =
        GenLogicalBlobName(model_load_op_conf.name(), model_load_conf->out());
    job_builder.AddOps(variable_op_parallel_conf, {new_var_op_conf, model_load_op_conf});
  }
}

void MakeModelSaveJob(
    const std::string& job_name, Job* job,
    const HashMap<std::string, OperatorConf>& var_op_name2op_conf,
    const HashMap<std::string, ParallelBlobConf>& var_op_name2parallel_blob_conf) {
  auto* flag_name2flag_value = job->mutable_job_conf()->mutable_flag_name2flag_value();
  (*flag_name2flag_value)["__is_user_function__"].set_at_bool(false);
  Global<InterUserJobInfo>::Get()->set_global_model_save_job_name(job_name);
  SetModelIoDefaultJobConf(job->mutable_job_conf(), job_name);
  JobBuilder job_builder(job);
  ParallelConf master_parallel_conf = GenParallelConfOfCpuZeroOnMaster();
  const OperatorConf tick_op_conf = GenTickOpConf("System-ModelSave-Tick");
  const OperatorConf foreign_input_op_conf = GenForeignInputOpConf(job_name, 65536);
  job_builder.AddOps(master_parallel_conf, {foreign_input_op_conf, tick_op_conf});
  if (var_op_name2op_conf.empty()) { return; }
  std::string prev_post_model_save_tick_lbn = GenLogicalBlobName(
      foreign_input_op_conf.name(), foreign_input_op_conf.foreign_input_conf().out());
  for (const auto& pair : var_op_name2op_conf) {
    const auto& var_op_name = pair.first;
    const OperatorConf& variable_op_conf = pair.second;
    const VariableOpConf& variable_conf = variable_op_conf.variable_conf();
    const auto& parallel_blob_conf = var_op_name2parallel_blob_conf.at(var_op_name);
    OperatorConf new_var_op_conf = CloneVariableOpConf(variable_op_conf);
    const std::string lbn = GetVariableLbn(variable_op_conf);
    OperatorConf model_save_op_conf{};
    model_save_op_conf.set_name("System-ModelSave-" + var_op_name);
    ModelSaveV2OpConf* model_save_conf = model_save_op_conf.mutable_model_save_v2_conf();
    model_save_conf->set_path(GenLogicalBlobName(foreign_input_op_conf.name(),
                                                 foreign_input_op_conf.foreign_input_conf().out()));
    *model_save_conf->mutable_in() = lbn;
    *model_save_conf->mutable_tick() = prev_post_model_save_tick_lbn;
    *model_save_conf->mutable_out() = "out";
    *model_save_conf->mutable_variable_op_name() = var_op_name;
    *model_save_conf->mutable_original_variable_conf() = variable_conf;
    prev_post_model_save_tick_lbn =
        GenLogicalBlobName(model_save_op_conf.name(), model_save_conf->out());
    job_builder.AddOps(parallel_blob_conf.parallel_conf(), {new_var_op_conf, model_save_op_conf});
  }
}

}  // namespace

void MakeModelIoV2Jobs(const std::vector<std::shared_ptr<Job>>& jobs,
                       const HashMap<std::string, ParallelBlobConf>& var_op_name2parallel_blob_conf,
                       const std::function<void(Job*)>& Handler) {
  HashMap<std::string, OperatorConf> var_op_name2op_conf;
  FilterVariableOps(jobs, &var_op_name2op_conf);
  {
    Job model_init_job;
    MakeModelInitJob("System-ModelInit", &model_init_job, var_op_name2op_conf,
                     var_op_name2parallel_blob_conf);
    Handler(&model_init_job);
  }
  {
    Job model_load_job;
    MakeModelLoadJob("System-ModelLoad", &model_load_job, var_op_name2op_conf,
                     var_op_name2parallel_blob_conf);
    Handler(&model_load_job);
  }
  {
    Job model_save_job;
    MakeModelSaveJob("System-ModelSave", &model_save_job, var_op_name2op_conf,
                     var_op_name2parallel_blob_conf);
    Handler(&model_save_job);
  }
}

}  // namespace oneflow
