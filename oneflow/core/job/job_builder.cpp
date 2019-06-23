#include "oneflow/core/job/job_builder.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

void SetBnValInOpTypeConf(PbMessage* pb_msg, const std::string& bn, const std::string& old_val,
                          const std::string& new_val) {
  const PbFd* fd = pb_msg->GetDescriptor()->FindFieldByName(bn);
  if (fd) {
    CHECK_EQ(GetValFromPbMessage<std::string>(*pb_msg, bn), old_val);
    SetValInPbMessage<std::string>(pb_msg, bn, new_val);
  } else {
    const std::pair<std::string, int32_t> prefix_idx = GenUnRepeatedBn(bn);
    CHECK_EQ(GetPbRpfFromPbMessage<std::string>(*pb_msg, prefix_idx.first).Get(prefix_idx.second),
             old_val);
    PbRpf<std::string>* rpf = MutPbRpfFromPbMessage<std::string>(pb_msg, prefix_idx.first);
    *rpf->Mutable(prefix_idx.second) = new_val;
  }
}

JobBuilder::JobBuilder(Job* job) : job_(job) {
  FOR_RANGE(int32_t, i, 0, job->net().op_size()) {
    CHECK(op_name2op_conf_.emplace(job->net().op(i).name(), job->mutable_net()->mutable_op(i))
              .second);
  }
  FOR_RANGE(int32_t, i, 0, job->placement().placement_group_size()) {
    auto* placemnt_group = job->mutable_placement()->mutable_placement_group(i);
    for (const auto& op_name : placemnt_group->op_set().op_name()) {
      CHECK(
          op_name2parallel_conf_.emplace(op_name, placemnt_group->mutable_parallel_conf()).second);
    }
  }
}

void JobBuilder::AddOps(const ParallelConf& parallel_conf,
                        const std::vector<OperatorConf>& op_confs) {
  auto* placemnt_group = job_->mutable_placement()->add_placement_group();
  *placemnt_group->mutable_parallel_conf() = parallel_conf;
  for (const auto& op_conf : op_confs) {
    CHECK(op_name2op_conf_.find(op_conf.name()) == op_name2op_conf_.end());
    OperatorConf* mut_op_conf = job_->mutable_net()->add_op();
    *mut_op_conf = op_conf;
    CHECK(op_name2op_conf_.emplace(op_conf.name(), mut_op_conf).second);
    placemnt_group->mutable_op_set()->add_op_name(op_conf.name());
    CHECK(op_name2parallel_conf_.emplace(op_conf.name(), placemnt_group->mutable_parallel_conf())
              .second);
  }
}

void JobBuilder::RemoveOp(const std::string &op_name) {
  // Update placement
  auto placement_group = job_->placement().placement_group();
  job_->mutable_placement()->clear_placement_group();
  for (const PlacementGroup &place : placement_group) {
    PlacementGroup p;
    OpNameSet *op_set = p.mutable_op_set();
    for (const std::string &name : place.op_set().op_name()) {
      if (name != op_name) {
        op_set->add_op_name(name);
      }
    }

    *(p.mutable_parallel_conf()) = place.parallel_conf();
    if (op_set->op_name().size() > 0) {
      *(job_->mutable_placement()->add_placement_group()) = p;
    }
  }
  // Update net
  DLNetConf net = job_->net();
  job_->mutable_net()->clear_op();
  job_->mutable_net()->clear_shared_model_group();
  for (const OperatorConf &op_conf : net.op()) {
    if (op_conf.name() != op_name) {
      *(job_->mutable_net()->add_op()) = op_conf;
    }
  }
  for (const OpNameSet &op_set : net.shared_model_group()) {
    OpNameSet s;
    for (const std::string &name : op_set.op_name()) {
      if (name != op_name) {
        s.add_op_name(name);
      }
    }
    if (s.op_name().size() >= 2) {
      *(job_->mutable_net()->add_shared_model_group()) = s;
    }
  }
  // Update Sbp
  auto *sbp_conf = job_->mutable_sbp_conf()
                       ->mutable_op_name2sbp_signature_conf();
  if (sbp_conf->count(op_name) > 0) {
    sbp_conf->erase(op_name);
  }
  // Update builder
  JobBuilder builder(job_);
  op_name2op_conf_.swap(builder.op_name2op_conf_);
  op_name2parallel_conf_.swap(builder.op_name2parallel_conf_);
}

void JobBuilder::MutOps(const std::vector<OperatorConf>& op_confs) const {
  for (const auto& op_conf : op_confs) { op_name2op_conf_.at(op_conf.name())->CopyFrom(op_conf); }
}

void JobBuilder::AddOrMutOps(const ParallelConf& parallel_conf,
                             const std::vector<OperatorConf>& op_confs) {
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

void JobBuilder::ForEachOperator(const std::function<void(const Operator&)>& Handler) const {
  for (const auto& pair : op_name2op_conf_) {
    DeviceType device_type = ParallelDesc(*op_name2parallel_conf_.at(pair.first)).device_type();
    std::shared_ptr<Operator> op = ConstructOp(*pair.second, device_type);
    Handler(*op);
  }
}

SbpParallel* JobBuilder::MutSbpParallel4Oba(const OpBlobArg& oba) const {
  auto* sbp_sig = &(*job_->mutable_sbp_conf()->mutable_op_name2sbp_signature_conf())[oba.op_name()];
  return &(*sbp_sig->mutable_bn_in_op2sbp_parallel())[oba.bn_in_op()];
}

void JobBuilder::BindIdenticalSbpOpBlobArgPair(const OpBlobArg& first, const OpBlobArg& second) {
  auto* pair = job_->mutable_helper()->mutable_identical_sbp_oba_pairs()->mutable_pair()->Add();
  *pair->mutable_first() = first;
  *pair->mutable_second() = second;
}

}  // namespace oneflow
