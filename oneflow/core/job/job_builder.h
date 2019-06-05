#ifndef ONEFLOW_CORE_JOB_JOB_CONF_BUILDER_H_
#define ONEFLOW_CORE_JOB_JOB_CONF_BUILDER_H_

#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/register/op_blob_arg.pb.h"

namespace oneflow {

void SetBnValInOpTypeConf(PbMessage* pb_msg, const std::string& bn, const std::string& old_val,
                          const std::string& new_val);

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

  const OperatorConf& OpConf4OpName(const std::string& op_name) const;
  void AddOps(const ParallelConf& parallel_conf, const std::vector<OperatorConf>& op_confs);
  void MutOps(const std::vector<OperatorConf>& op_confs) const;
  void AddOrMutOps(const ParallelConf& parallel_conf, const std::vector<OperatorConf>& op_confs);
  SbpParallel* MutSbpParallel4Oba(const OpBlobArg& oba) const;
  void BindIdenticalSbpOpBlobArgPair(const OpBlobArg& first, const OpBlobArg& second);

  void ForEachOperator(const std::function<void(const Operator&)>& Handler) const;

 private:
  Job* job_;
  HashMap<std::string, OperatorConf*> op_name2op_conf_;
  HashMap<std::string, ParallelConf*> op_name2parallel_conf_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_JOB_CONF_BUILDER_H_
