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
  SbpConf* mutable_sbp_conf() { return job_->mutable_sbp_conf(); }

  const OperatorConf& OpConf4OpName(const std::string& op_name) const;
  const ParallelConf& ParallelConf4OpName(const std::string& op_name) const;
  const ParallelConf& ParallelConf4Lbi(const LogicalBlobId& lbi) const;
  void AddOps(const ParallelConf& parallel_conf, const std::vector<OperatorConf>& op_confs);
  void MutOpsOnlyOnce(const std::vector<OperatorConf>& op_confs);
  void MutParallelConfOnlyOnce(const std::string& op_name, const ParallelConf& parallel_conf);
  void AddOrMutOpsOnlyOnce(const ParallelConf& parallel_conf,
                           const std::vector<OperatorConf>& op_confs);
  void DelOps(const std::vector<OperatorConf>& op_confs);
  SbpParallel* MutSbpParallel4Oba(const OpBlobArg& oba) const;
  void BindIdenticalSbpOpBlobArgPair(const OpBlobArg& first, const OpBlobArg& second);

  void ForEachOperator(const std::function<void(const Operator&)>& Handler) const;

 private:
  PlacementGroup* FindPlacementGroup(const std::string& op_name) const;

  Job* job_;
  HashMap<std::string, OperatorConf*> op_name2op_conf_;
  HashMap<std::string, ParallelConf*> op_name2parallel_conf_;
  HashMap<LogicalBlobId, ParallelConf*> lbi2blob_parallel_conf_;
  HashSet<std::string> modified_op_conf_op_names_;
  HashSet<std::string> modified_parallel_conf_op_names_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_JOB_CONF_BUILDER_H_
