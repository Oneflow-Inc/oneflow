#include "oneflow/core/job/job_conf_builder.h"

namespace oneflow {

void JobConfBuilder::AddOps(const ParallelConf& parallel_conf,
                            const std::vector<OperatorConf>& op_confs) const {
  auto* placemnt_group = job_conf_->mutable_placement()->add_placement_group();
  *placemnt_group->mutable_parallel_conf() = parallel_conf;
  for (const auto& op_conf : op_confs) {
    *job_conf_->mutable_net()->add_op() = op_conf;
    placemnt_group->mutable_op_set()->add_op_name(op_conf.name());
  }
}

}  // namespace oneflow
