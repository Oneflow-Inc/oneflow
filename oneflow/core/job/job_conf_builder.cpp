#include "oneflow/core/job/job_conf_builder.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

JobConfBuilder::JobConfBuilder(JobConf1* job_conf) : job_conf_(job_conf) {
  FOR_RANGE(int, i, 0, job_conf->net().op_size()) {
    CHECK(op_name2op_conf_
              .emplace(job_conf->net().op(i).name(), job_conf->mutable_net()->mutable_op(i))
              .second);
  }
}

void JobConfBuilder::AddOps(const ParallelConf& parallel_conf,
                            const std::vector<OperatorConf>& op_confs) const {
  auto* placemnt_group = job_conf_->mutable_placement()->add_placement_group();
  *placemnt_group->mutable_parallel_conf() = parallel_conf;
  for (const auto& op_conf : op_confs) {
    CHECK(op_name2op_conf_.find(op_conf.name()) == op_name2op_conf_.end());
    *job_conf_->mutable_net()->add_op() = op_conf;
    placemnt_group->mutable_op_set()->add_op_name(op_conf.name());
  }
}

void JobConfBuilder::MutOps(const std::vector<OperatorConf>& op_confs) const {
  for (const auto& op_conf : op_confs) { *op_name2op_conf_.at(op_conf.name()) = op_conf; }
}

void JobConfBuilder::AddOrMutOps(const ParallelConf& parallel_conf,
                                 const std::vector<OperatorConf>& op_confs) const {
  std::vector<OperatorConf> add_ops;
  std::vector<OperatorConf> mut_ops;
  for (const auto& op_conf : op_confs) {
    if (op_name2op_conf_.find(op_conf.name()) == op_name2op_conf_.end()) {
      add_ops.push_back(op_conf);
    } else {
      mut_ops.push_back(op_conf);
    }
  }
  AddOps(parallel_conf, add_ops);
  MutOps(mut_ops);
}

}  // namespace oneflow
