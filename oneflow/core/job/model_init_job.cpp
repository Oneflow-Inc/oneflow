#include "oneflow/core/job/model_init_job.h"
#include "oneflow/core/operator/interface_op_util.h"

namespace oneflow {

bool CompareVariableOpconf(const VariableOpConf& lhs, const VariableOpConf& rhs) {
  if (lhs.has_random_seed() && rhs.has_random_seed()) {
    CHECK_EQ(lhs.random_seed(), rhs.random_seed());
  }
  VariableOpConf var_conf_a(lhs);
  VariableOpConf var_conf_b(rhs);
  var_conf_a.clear_tick();
  var_conf_b.clear_tick();
  var_conf_a.clear_out();
  var_conf_b.clear_out();
  return PbMd().Equals(var_conf_a, var_conf_b);
}

void FilterVariableOps(std::vector<Job>& jobs,
                       HashMap<std::string, OperatorConf>* var_op_name2op_conf) {
  FOR_RANGE(int64_t, job_id, 0, jobs.size()) {
    const JobBuilder job_builder(&jobs.at(job_id));
    for (const OperatorConf& op_conf : jobs.at(job_id).net().op()) {
      if (op_conf.has_variable_conf()) {
        if (var_op_name2op_conf->find(op_conf.name()) == var_op_name2op_conf->end()) {
          CHECK(var_op_name2op_conf->emplace(op_conf.name(), op_conf).second);
        } else {
          CHECK(CompareVariableOpconf(var_op_name2op_conf->at(op_conf.name()).variable_conf(),
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
  JobBuilder job_builder(job);
  ParallelConf master_parallel_conf;
  master_parallel_conf.set_policy(kDataParallel);
  master_parallel_conf.add_device_name("0:cpu:0");
  auto* job_conf = job->mutable_job_conf();

  // there is always a tick op in model save job, so we can ignore the case with no variable
  OperatorConf tick_op_conf;
  tick_op_conf.set_name("System-ModelInit-tick");
  tick_op_conf.mutable_tick_conf()->set_out("out");
  job_builder.AddOps(master_parallel_conf, {tick_op_conf});

  for (const auto& pair : var_op_name2op_conf) {
    const auto& var_op_name = pair.first;
    const OperatorConf& variable_op_conf = pair.second;
    const auto& variable_op_parallel_blob_conf = var_op_name2parallel_blob_conf.at(var_op_name);
    const ParallelConf& variable_op_parallel_conf = variable_op_parallel_blob_conf.parallel_conf();
    CHECK_NE(variable_op_conf.variable_conf().data_type(), DataType::kInvalidDataType);
    CHECK(variable_op_conf.variable_conf().has_initializer());

    OperatorConf model_init_op_conf;
    model_init_op_conf.set_name("System-Init-" + var_op_name + "-ModelInit");
    ModelInitOpConf* model_init_conf = model_init_op_conf.mutable_model_init_conf();
    model_init_conf->set_variable_op_name(var_op_name);
    *model_init_conf->mutable_original_variable_conf() = variable_op_conf.variable_conf();
    model_init_conf->set_out("out");
    job_builder.AddOps(master_parallel_conf, {model_init_op_conf});

    OperatorConf output_op_conf;
    output_op_conf.set_name(var_op_name);
    auto* output_conf = output_op_conf.mutable_output_conf();
    output_conf->set_in(model_init_op_conf.name() + "/out");
    output_conf->set_out("out");
    InterfaceOpUtil::InitBlobConf(output_conf->mutable_blob_conf(), variable_op_parallel_blob_conf);
    job_builder.AddOps(variable_op_parallel_conf, {output_op_conf});
    job_conf->add_arg_op_name(output_op_conf.name());
  }

  const std::string global_model_init_job_name = "System-ModelInit";
  job_conf->set_job_name(global_model_init_job_name);
  job_conf->mutable_predict_conf();
  job_conf->set_piece_size(1);
  job_conf->set_data_part_num(1);
  job_conf->set_total_batch_num(1);
  Global<InterUserJobInfo>::Get()->set_global_model_init_job_name(global_model_init_job_name);
}

}  // namespace oneflow
