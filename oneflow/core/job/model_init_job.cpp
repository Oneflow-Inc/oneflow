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

void MakeModelInitJob(Job* job, const std::vector<std::pair<OperatorConf, ParallelConf>>&
                                    variable_op_confs_and_parallel_confs) {
  JobBuilder job_builder(job);
  auto* job_conf = job->mutable_job_conf();
  ParallelConf master_parallel_conf;
  master_parallel_conf.set_policy(kDataParallel);
  master_parallel_conf.add_device_name("0:cpu:0");
  for (const auto& variable_op_conf_tuple : variable_op_confs_and_parallel_confs) {
    OperatorConf variable_op_conf;
    ParallelConf variable_op_parallel_conf;
    std::tie(variable_op_conf, variable_op_parallel_conf) = variable_op_conf_tuple;
    job_conf->add_arg_op_name(variable_op_conf.name());

    OperatorConf model_init_op_conf;
    const std::string model_init_op_name = variable_op_conf.name() + "_model_init";
    model_init_op_conf.set_name(model_init_op_name);
    ModelInitOpConf* model_init_conf = model_init_op_conf.mutable_model_init_conf();
    model_init_conf->set_variable_op_name(variable_op_conf.name());
    *model_init_conf->mutable_variable_conf() = variable_op_conf.variable_conf();
    model_init_conf->set_out("out");
    job_builder.AddOps(master_parallel_conf, {model_init_op_conf});

    OperatorConf assign_op_conf;
    assign_op_conf.set_name(variable_op_conf.name() + "_model_init_assign");
    AssignOpConf* assign_conf = assign_op_conf.mutable_assign_conf();
    assign_conf->set_x(variable_op_conf.name() + "/" + variable_op_conf.variable_conf().out());
    assign_conf->set_value(model_init_op_name + "/out");
    job_builder.AddOps(variable_op_parallel_conf, {assign_op_conf});
  }
  const std::string model_init_job_name = "ModelInitJob";
  job_conf->set_job_name(model_init_job_name);
  job_conf->mutable_predict_conf();
  job_conf->set_piece_size(1);
  job_conf->set_data_part_num(1);
  job_conf->set_total_batch_num(1);
  Global<InterUserJobInfo>::Get()->set_model_init_job_name(model_init_job_name);
}

}  // namespace oneflow
