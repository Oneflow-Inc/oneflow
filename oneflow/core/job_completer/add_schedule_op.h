#ifndef ONEFLOW_CORE_JOB_COMPLETER_ADD_SCHEDULE_OP_H_
#define ONEFLOW_CORE_JOB_COMPLETER_ADD_SCHEDULE_OP_H_

namespace oneflow {

class OpGraph;
class Job;

std::string AddScheduleOp(JobBuilder* job_builder,
                          const NormalModelUpdateOpUserConf& model_update_conf,
                          const std::string& train_step_lbn, const std::string& op_name,
                          const float learning_rate);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COMPLETER_ADD_SCHEDULE_OP_H_
