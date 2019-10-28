#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/job_completer/add_schedule_op.h"
#include "oneflow/core/job/job.pb.h"

namespace oneflow {

void AutoLearningRate(const OpGraph& op_graph, Job* job) {
  JobBuilder job_builder(job);
  const TrainConf& train_conf = job->job_conf().train_conf();
  if (!train_conf.has_primary_lr_lbn()) {
    CHECK(train_conf.has_primary_lr());
    const std::string lbn =
        AddScheduleOp(&job_builder, train_conf.model_update_conf(), train_conf.train_step_lbn(),
                      "System-Train-PrimaryLearningRate-Scheduler", train_conf.primary_lr());
    job->mutable_job_conf()->mutable_train_conf()->set_primary_lr_lbn(lbn);
  }
  if (!train_conf.has_secondary_lr_lbn()) {
    if (train_conf.has_secondary_lr()) {
      const std::string lbn =
          AddScheduleOp(&job_builder, train_conf.model_update_conf(), train_conf.train_step_lbn(),
                        "System-Train-SecondaryLearningRate-Scheduler", train_conf.secondary_lr());
      job->mutable_job_conf()->mutable_train_conf()->set_secondary_lr_lbn(lbn);
    } else {
      job->mutable_job_conf()->mutable_train_conf()->set_secondary_lr_lbn(
          train_conf.primary_lr_lbn());
    }
  }
}

}  // namespace oneflow
