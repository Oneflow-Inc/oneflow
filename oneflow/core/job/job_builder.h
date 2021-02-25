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
#ifndef ONEFLOW_CORE_JOB_JOB_CONF_BUILDER_H_
#define ONEFLOW_CORE_JOB_JOB_CONF_BUILDER_H_

#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/register/op_blob_arg.pb.h"

namespace oneflow {

const static std::string kProducedLbi2ConsumedDiffLbi = "produced_lbi2consumed_diff_lbi";

std::function<const ParallelConf*(const std::string&)> MakeGetterParallelConf4OpName(
    const Placement& placement);

class SbpParallel;
class LogicalBlobId;
class Operator;

class JobBuilder final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(JobBuilder);
  explicit JobBuilder(Job* job);
  ~JobBuilder() = default;

  const Job& job() const { return *job_; }
  JobHelperConf* mutable_helper() { return job_->mutable_helper(); }
  JobParallelViewConf* mutable_job_parallel_view_conf() {
    return job_->mutable_job_parallel_view_conf();
  }

  const OperatorConf& OpConf4OpName(const std::string& op_name) const;
  OperatorConf* MutableOpConf4OpName(const std::string& op_name);

  Maybe<void> AddOp(const ParallelConf& parallel_conf, const OperatorConf& op_conf);
  void AddOps(const ParallelConf& parallel_conf, const std::vector<OperatorConf>& op_confs);
  Maybe<void> MutOpOnlyOnce(const OperatorConf& op_conf);
  void MutOpsOnlyOnce(const std::vector<OperatorConf>& op_confs);
  void MutParallelConfOnlyOnce(const std::string& op_name, const ParallelConf& parallel_conf);
  void AddOrMutOpsOnlyOnce(const ParallelConf& parallel_conf,
                           const std::vector<OperatorConf>& op_confs);

  void RemoveOpByName(const std::string& op_name);
  void RemoveOpByName(const std::unordered_set<std::string>& removing_names);
  void DelOps(const std::vector<std::string>& op_names);
  void DelOps(const std::vector<OperatorConf>& op_confs);

  SbpParallel* MutSbpParallel4Oba(const OpBlobArg& oba) const;
  void BindIdenticalSbpOpBlobArgPair(const OpBlobArg& first, const OpBlobArg& second);

  void ForEachOperator(const std::function<void(const Operator&)>& Handler) const;

  const ParallelConf& ParallelConf4Lbi(const LogicalBlobId& lbi) const;
  const ParallelConf& ParallelConf4OpName(const std::string& op_name) const;
  void AddParallelConf4OpName(const std::string& op_name, const ParallelConf& parallel_conf);

  const SbpSignature& SbpSignature4OpName(const std::string& op_name) const;
  void AddSbpSignature4OpName(const std::string& op_name, const SbpSignature& sbp_signature);

  const OpTimeShape& TimeShape4OpName(const std::string& op_name) const;
  void AddTimeShape4OpName(const std::string& op_name, const OpTimeShape& time_shape);

 private:
  PlacementGroup* FindPlacementGroup(const std::string& op_name) const;

  Job* job_;
  HashMap<std::string, OperatorConf*> op_name2op_conf_;
  HashMap<std::string, ParallelConf*> op_name2parallel_conf_;
  HashMap<LogicalBlobId, ParallelConf*> lbi2blob_parallel_conf_;
  HashSet<std::string> modified_op_conf_op_names_;
  HashSet<std::string> modified_parallel_conf_op_names_;

  HashMap<std::string, SbpSignature*> op_name2sbp_signature_conf_;
  HashMap<std::string, OpTimeShape*> op_name2time_shapes_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_JOB_CONF_BUILDER_H_
