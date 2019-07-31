#include "oneflow/core/job/model_init_job.h"

namespace oneflow {

void FilterVariableOps(
    std::vector<Job>& jobs,
    std::vector<std::pair<OperatorConf, ParallelConf>>* variable_op_confs_and_parallel_confs) {
  FOR_RANGE(int64_t, job_id, 0, jobs.size()) {
    const JobBuilder job_builder(&jobs.at(job_id));
    for (const OperatorConf& op_conf : jobs.at(job_id).net().op()) {
      if (op_conf.has_variable_conf()) {
        variable_op_confs_and_parallel_confs->push_back(
            std::make_pair(op_conf, job_builder.ParallelConf4OpName(op_conf.name())));
      }
    }
  }
}

void MakeModelInitJob(const std::string& job_name, Job* job,
                      const std::vector<std::pair<OperatorConf, ParallelConf>>&
                          variable_op_confs_and_parallel_confs) {
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

  for (const auto& variable_op_conf_tuple : variable_op_confs_and_parallel_confs) {
    OperatorConf variable_op_conf;
    ParallelConf variable_op_parallel_conf;
    std::tie(variable_op_conf, variable_op_parallel_conf) = variable_op_conf_tuple;
    CHECK_NE(variable_op_conf.variable_conf().data_type(), DataType::kInvalidDataType);
    CHECK(variable_op_conf.variable_conf().has_initializer());

    OperatorConf model_init_op_conf;
    model_init_op_conf.set_name("System-Init-" + variable_op_conf.name() + "_ModelInit");
    ModelInitOpConf* model_init_conf = model_init_op_conf.mutable_model_init_conf();
    model_init_conf->set_variable_op_name(variable_op_conf.name());
    *model_init_conf->mutable_original_variable_conf() = variable_op_conf.variable_conf();
    model_init_conf->set_out("out");
    model_init_conf->set_random_seed(NewRandomSeed());
    job_builder.AddOps(master_parallel_conf, {model_init_op_conf});

    OperatorConf output_op_conf;
    output_op_conf.set_name(variable_op_conf.name());
    auto* output_conf = output_op_conf.mutable_output_conf();
    output_conf->set_in(model_init_op_conf.name() + "/out");
    output_conf->set_out("out");
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
