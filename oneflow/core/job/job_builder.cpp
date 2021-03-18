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
#include "oneflow/core/job/job_builder.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

namespace {

int64_t GetParallelHierarchyNumAxes(
    const HashMap<std::string, ParallelConf*>& op_name2parallel_conf, const std::string& op_name) {
  const auto& it = op_name2parallel_conf.find(op_name);
  CHECK(it != op_name2parallel_conf.end());
  if (!it->second->has_hierarchy()) {
    return 1;
  } else {
    return it->second->hierarchy().dim_size();
  }
}

void SetParallelDistributionSignature4Oba(Job* job,
                                          HashMap<std::string, ParallelDistributionSignature*>*
                                              op_name2parallel_distribution_signature_map,
                                          const OpBlobArg& oba,
                                          const ParallelDistribution& parallel_distribution) {
  auto* parallel_distribution_sig =
      &(*job->mutable_job_parallel_view_conf()
             ->mutable_op_name2parallel_distribution_signature_conf())[oba.op_name()];
  (*parallel_distribution_sig->mutable_bn_in_op2parallel_distribution())[oba.bn_in_op()] =
      parallel_distribution;
  auto* op_name2parallel_distribution_signature_conf =
      job->mutable_job_parallel_view_conf()->mutable_op_name2parallel_distribution_signature_conf();
  (*op_name2parallel_distribution_signature_map)[oba.op_name()] =
      &(*op_name2parallel_distribution_signature_conf)[oba.op_name()];
}

void SetSbpSignature4Oba(Job* job, const OpBlobArg& oba, const SbpParallel& sbp_parallel) {
  auto* sbp_sig = &(
      *job->mutable_job_parallel_view_conf()->mutable_op_name2sbp_signature_conf())[oba.op_name()];
  (*sbp_sig->mutable_bn_in_op2sbp_parallel())[oba.bn_in_op()] = sbp_parallel;
}

void AddOrSetParallelDistributionSignature4OpName(
    Job* job,
    HashMap<std::string, ParallelDistributionSignature*>*
        op_name2parallel_distribution_signature_map,
    const std::string& op_name,
    const ParallelDistributionSignature& parallel_distribution_signature) {
  const auto& it = op_name2parallel_distribution_signature_map->find(op_name);
  if (it != op_name2parallel_distribution_signature_map->end()) {
    *(it->second) = parallel_distribution_signature;
  } else {
    auto* op_name2parallel_distribution_signature_conf =
        job->mutable_job_parallel_view_conf()
            ->mutable_op_name2parallel_distribution_signature_conf();
    (*op_name2parallel_distribution_signature_conf)[op_name] = parallel_distribution_signature;
    op_name2parallel_distribution_signature_map->emplace(
        op_name, &(*op_name2parallel_distribution_signature_conf)[op_name]);
  }
}

void AddOrSetSbpSignature4OpName(Job* job, const std::string& op_name,
                                 const SbpSignature& sbp_signature) {
  auto* op_name2sbp_signature_conf =
      job->mutable_job_parallel_view_conf()->mutable_op_name2sbp_signature_conf();
  (*op_name2sbp_signature_conf)[op_name] = sbp_signature;
}

}  // namespace

std::function<const ParallelConf*(const std::string&)> MakeGetterParallelConf4OpName(
    const Placement& placement) {
  auto op_name2parallel_conf = std::make_shared<HashMap<std::string, const ParallelConf*>>();
  for (const auto& placement_group : placement.placement_group()) {
    for (const std::string& op_name : placement_group.op_set().op_name()) {
      const ParallelConf* parallel_conf = &placement_group.parallel_conf();
      CHECK(op_name2parallel_conf->emplace(op_name, parallel_conf).second)
          << "op_name: " << op_name;
    }
  }
  return [op_name2parallel_conf](const std::string& op_name) {
    return op_name2parallel_conf->at(op_name);
  };
}

JobBuilder::JobBuilder(Job* job) : job_(job) {
  FOR_RANGE(int32_t, i, 0, job->net().op_size()) {
    CHECK(op_name2op_conf_.emplace(job->net().op(i).name(), job->mutable_net()->mutable_op(i))
              .second);
  }
  bool all_ops_1d_hierarchy = true;
  FOR_RANGE(int32_t, i, 0, job->placement().placement_group_size()) {
    auto* placemnt_group = job->mutable_placement()->mutable_placement_group(i);
    if (placemnt_group->parallel_conf().has_hierarchy()
        && placemnt_group->parallel_conf().hierarchy().dim_size() > 1) {
      all_ops_1d_hierarchy = false;
    }
  }
  auto* job_parallel_view_conf = job->mutable_job_parallel_view_conf();
  for (auto& pair :
       *(job_parallel_view_conf->mutable_op_name2parallel_distribution_signature_conf())) {
    op_name2parallel_distribution_signature_conf_.emplace(pair.first, &pair.second);
  }
  if (all_ops_1d_hierarchy) {
    CHECK_EQ(job_parallel_view_conf->op_name2sbp_signature_conf_size(),
             job_parallel_view_conf->op_name2parallel_distribution_signature_conf_size());
    for (const auto& pair :
         job_parallel_view_conf->op_name2parallel_distribution_signature_conf()) {
      const auto& op_name2sbp_sig = job_parallel_view_conf->op_name2sbp_signature_conf();
      const auto it = op_name2sbp_sig.find(pair.first);
      CHECK(it != op_name2sbp_sig.end());
      CheckSbpSignatureAndParallelDistributionEquals(it->second, pair.second);
    }
  }
  FOR_RANGE(int32_t, i, 0, job->placement().blob_placement_group_size()) {
    auto* blob_pg = job->mutable_placement()->mutable_blob_placement_group(i);
    for (const auto& lbi : blob_pg->lbi()) {
      CHECK(lbi2blob_parallel_conf_.emplace(lbi, blob_pg->mutable_parallel_conf()).second);
    }
  }
  for (auto& placement_group : *job->mutable_placement()->mutable_placement_group()) {
    if (placement_group.op_set().op_name().empty()) { continue; }
    const ParallelConf& parallel_conf = placement_group.parallel_conf();
    auto it = parallel_conf2placement_group_.find(parallel_conf);
    if (it == parallel_conf2placement_group_.end()) {
      parallel_conf2placement_group_.emplace(parallel_conf, &placement_group);
      for (const auto& op_name : placement_group.op_set().op_name()) {
        CHECK(op_name2parallel_conf_.emplace(op_name, placement_group.mutable_parallel_conf())
                  .second);
      }
    } else {
      PlacementGroup* existing_placement_group = it->second;
      for (const auto& op_name : placement_group.op_set().op_name()) {
        *existing_placement_group->mutable_op_set()->mutable_op_name()->Add() = op_name;
        CHECK(op_name2parallel_conf_
                  .emplace(op_name, existing_placement_group->mutable_parallel_conf())
                  .second);
      }
      placement_group.mutable_op_set()->mutable_op_name()->Clear();
    }
  }
}

OperatorConf* JobBuilder::MutableOpConf4OpName(const std::string& op_name) {
  const auto& it = op_name2op_conf_.find(op_name);
  CHECK(it != op_name2op_conf_.end());
  return it->second;
}

const OperatorConf& JobBuilder::OpConf4OpName(const std::string& op_name) const {
  return *op_name2op_conf_.at(op_name);
}

const ParallelConf& JobBuilder::ParallelConf4Lbi(const LogicalBlobId& lbi) const {
  const auto& iter = lbi2blob_parallel_conf_.find(lbi);
  if (iter != lbi2blob_parallel_conf_.end()) { return *iter->second; }
  return ParallelConf4OpName(lbi.op_name());
}

Maybe<void> JobBuilder::AddOp(const ParallelConf& parallel_conf, const OperatorConf& op_conf) {
  CHECK_OR_RETURN(op_name2op_conf_.find(op_conf.name()) == op_name2op_conf_.end());
  OperatorConf* mut_op_conf = job_->mutable_net()->add_op();
  *mut_op_conf = op_conf;
  CHECK_OR_RETURN(op_name2op_conf_.emplace(op_conf.name(), mut_op_conf).second);
  AddOpNamesToPlacementGroup({op_conf.name()}, parallel_conf);
  return Maybe<void>::Ok();
}

void JobBuilder::AddOps(const ParallelConf& parallel_conf,
                        const std::vector<OperatorConf>& op_confs) {
  if (op_confs.empty()) { return; }
  std::vector<std::string> op_names;
  op_names.reserve(op_confs.size());
  for (const auto& op_conf : op_confs) {
    CHECK(op_name2op_conf_.find(op_conf.name()) == op_name2op_conf_.end());
    OperatorConf* mut_op_conf = job_->mutable_net()->add_op();
    *mut_op_conf = op_conf;
    CHECK(op_name2op_conf_.emplace(op_conf.name(), mut_op_conf).second);
    op_names.emplace_back(op_conf.name());
  }
  AddOpNamesToPlacementGroup(op_names, parallel_conf);
}

void JobBuilder::AddOpNamesToPlacementGroup(const std::vector<std::string>& op_names,
                                            const ParallelConf& parallel_conf) {
  PlacementGroup* placement_group = nullptr;
  auto it = parallel_conf2placement_group_.find(parallel_conf);
  if (it != parallel_conf2placement_group_.end()) {
    placement_group = it->second;
  } else {
    placement_group = job_->mutable_placement()->add_placement_group();
    *placement_group->mutable_parallel_conf() = parallel_conf;
    parallel_conf2placement_group_.emplace(parallel_conf, placement_group);
  }
  for (const auto& op_name : op_names) {
    placement_group->mutable_op_set()->add_op_name(op_name);
    CHECK(op_name2parallel_conf_.emplace(op_name, placement_group->mutable_parallel_conf()).second);
  }
}

void JobBuilder::MutParallelConfOnlyOnce(const std::string& op_name,
                                         const ParallelConf& parallel_conf) {
  CHECK(modified_parallel_conf_op_names_.emplace(op_name).second);
  const auto& parallel_conf_it = op_name2parallel_conf_.find(op_name);
  CHECK(parallel_conf_it != op_name2parallel_conf_.end());
  auto old_placement_group_it = parallel_conf2placement_group_.find(*parallel_conf_it->second);
  CHECK(old_placement_group_it != parallel_conf2placement_group_.end());
  op_name2parallel_conf_.erase(parallel_conf_it);
  Erase<PbRpf<std::string>>(*old_placement_group_it->second->mutable_op_set()->mutable_op_name(),
                            [&](const std::string& x) { return x == op_name; });
  AddOpNamesToPlacementGroup({op_name}, parallel_conf);
}

void JobBuilder::RemoveOpByName(const std::string& op_name) {
  RemoveOpByName(std::unordered_set<std::string>{op_name});
}

void JobBuilder::RemoveOpByName(const std::unordered_set<std::string>& removing_names) {
  // Update net
  DLNetConf net = job_->net();
  job_->mutable_net()->clear_op();
  for (const OperatorConf& op_conf : net.op()) {
    if (removing_names.count(op_conf.name()) == 0) { *(job_->mutable_net()->add_op()) = op_conf; }
  }
  // Update placement
  auto placement_group = job_->placement().placement_group();
  job_->mutable_placement()->clear_placement_group();
  for (const PlacementGroup& place : placement_group) {
    PlacementGroup p;
    OpNameSet* op_set = p.mutable_op_set();
    for (const std::string& name : place.op_set().op_name()) {
      if (removing_names.count(name) == 0) { op_set->add_op_name(name); }
    }

    *(p.mutable_parallel_conf()) = place.parallel_conf();
    if (op_set->op_name().size() > 0) { *(job_->mutable_placement()->add_placement_group()) = p; }
  }

  auto* op_name2sbp_signature_conf =
      job_->mutable_job_parallel_view_conf()->mutable_op_name2sbp_signature_conf();
  auto* op_name2parallel_distribution_signature_conf =
      job_->mutable_job_parallel_view_conf()
          ->mutable_op_name2parallel_distribution_signature_conf();
  for (const std::string& op_name : removing_names) {
    // Update ParallelDistribution, Sbp
    if (op_name2parallel_distribution_signature_conf->count(op_name) > 0) {
      op_name2parallel_distribution_signature_conf->erase(op_name);
      if (GetParallelHierarchyNumAxes(op_name2parallel_conf_, op_name) == 1) {
        CHECK(op_name2sbp_signature_conf->count(op_name) > 0);
        op_name2sbp_signature_conf->erase(op_name);
      }
    }
  }
  // Update builder
  JobBuilder builder(job_);
  op_name2op_conf_.swap(builder.op_name2op_conf_);
  op_name2parallel_conf_.swap(builder.op_name2parallel_conf_);
  op_name2parallel_distribution_signature_conf_.swap(
      builder.op_name2parallel_distribution_signature_conf_);
  parallel_conf2placement_group_.swap(builder.parallel_conf2placement_group_);
}

void JobBuilder::DelOps(const std::vector<std::string>& op_names) {
  std::unordered_set<std::string> removing_names;
  for (const auto& op_name : op_names) { removing_names.insert(op_name); }
  RemoveOpByName(removing_names);
}

void JobBuilder::DelOps(const std::vector<OperatorConf>& op_confs) {
  std::unordered_set<std::string> removing_names;
  for (const auto& op_conf : op_confs) { removing_names.insert(op_conf.name()); }
  RemoveOpByName(removing_names);
}

Maybe<void> JobBuilder::MutOpOnlyOnce(const OperatorConf& op_conf) {
  CHECK_OR_RETURN(modified_op_conf_op_names_.emplace(op_conf.name()).second);
  op_name2op_conf_.at(op_conf.name())->CopyFrom(op_conf);
  return Maybe<void>::Ok();
}

void JobBuilder::MutOpsOnlyOnce(const std::vector<OperatorConf>& op_confs) {
  for (const auto& op_conf : op_confs) {
    CHECK(modified_op_conf_op_names_.emplace(op_conf.name()).second);
    op_name2op_conf_.at(op_conf.name())->CopyFrom(op_conf);
  }
}

void JobBuilder::AddOrMutOpsOnlyOnce(const ParallelConf& parallel_conf,
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
  MutOpsOnlyOnce(mut_ops);
}

void JobBuilder::ForEachOperator(const std::function<void(const Operator&)>& Handler) const {
  for (const auto& pair : op_name2op_conf_) {
    auto it = op_name2parallel_conf_.find(pair.first);
    CHECK(it != op_name2parallel_conf_.end()) << "op_name: " << pair.first;
    DeviceType device_type = ParallelDesc(*it->second).device_type();
    std::shared_ptr<Operator> op = ConstructOp(*pair.second, device_type);
    Handler(*op);
  }
}

const ParallelConf& JobBuilder::ParallelConf4OpName(const std::string& op_name) const {
  const auto& iter = op_name2parallel_conf_.find(op_name);
  CHECK(iter != op_name2parallel_conf_.end());
  return *iter->second;
}

SbpParallel* JobBuilder::MutSbpParallel4Oba(const OpBlobArg& oba) const {
  // TODO(guoran): rm this func
  auto* sbp_sig = &(
      *job_->mutable_job_parallel_view_conf()->mutable_op_name2sbp_signature_conf())[oba.op_name()];
  return &(*sbp_sig->mutable_bn_in_op2sbp_parallel())[oba.bn_in_op()];
}

void JobBuilder::SetSbpParallel4Oba(const OpBlobArg& oba, const SbpParallel& sbp_parallel) {
  CHECK_EQ(GetParallelHierarchyNumAxes(op_name2parallel_conf_, oba.op_name()), 1);
  SetSbpSignature4Oba(job_, oba, sbp_parallel);
  ParallelDistribution parallel_distribution;
  *parallel_distribution.add_sbp_parallel() = sbp_parallel;
  SetParallelDistributionSignature4Oba(job_, &op_name2parallel_distribution_signature_conf_, oba,
                                       parallel_distribution);
}

void JobBuilder::SetParallelDistribution4Oba(const OpBlobArg& oba,
                                             const ParallelDistribution& parallel_distribution) {
  SetParallelDistributionSignature4Oba(job_, &op_name2parallel_distribution_signature_conf_, oba,
                                       parallel_distribution);
  if (GetParallelHierarchyNumAxes(op_name2parallel_conf_, oba.op_name()) == 1) {
    SetSbpSignature4Oba(job_, oba, parallel_distribution.sbp_parallel(0));
  }
}

const SbpSignature JobBuilder::SbpSignature4OpName(const std::string& op_name) const {
  CHECK_EQ(GetParallelHierarchyNumAxes(op_name2parallel_conf_, op_name), 1);
  const auto& it = op_name2parallel_distribution_signature_conf_.find(op_name);
  CHECK(it != op_name2parallel_distribution_signature_conf_.end());

  SbpSignature sbp_sig_conf;
  ParallelDistributionSignatureToSbpSignature(*it->second, &sbp_sig_conf);
  return sbp_sig_conf;
}

void JobBuilder::AddSbpSignature4OpName(const std::string& op_name,
                                        const SbpSignature& sbp_signature) {
  ParallelDistributionSignature parallel_distribution_signature;
  SbpSignatureToParallelDistributionSignature(sbp_signature, &parallel_distribution_signature);
  AddOrSetParallelDistributionSignature4OpName(job_, &op_name2parallel_distribution_signature_conf_,
                                               op_name, parallel_distribution_signature);
  CHECK_EQ(GetParallelHierarchyNumAxes(op_name2parallel_conf_, op_name), 1);
  AddOrSetSbpSignature4OpName(job_, op_name, sbp_signature);
}

const ParallelDistributionSignature& JobBuilder::ParallelDistributionSignature4OpName(
    const std::string& op_name) const {
  const auto& it = op_name2parallel_distribution_signature_conf_.find(op_name);
  CHECK(it != op_name2parallel_distribution_signature_conf_.end());
  return *(it->second);
}

void JobBuilder::AddParallelDistributionSignature4OpName(
    const std::string& op_name,
    const ParallelDistributionSignature& parallel_distribution_signature) {
  AddOrSetParallelDistributionSignature4OpName(job_, &op_name2parallel_distribution_signature_conf_,
                                               op_name, parallel_distribution_signature);
  if (GetParallelHierarchyNumAxes(op_name2parallel_conf_, op_name) == 1) {
    SbpSignature sbp_signature;
    ParallelDistributionSignatureToSbpSignature(parallel_distribution_signature, &sbp_signature);
    AddOrSetSbpSignature4OpName(job_, op_name, sbp_signature);
  }
}

}  // namespace oneflow
