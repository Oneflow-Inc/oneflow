#ifndef ONEFLOW_CORE_JOB_SCOPE_H_
#define ONEFLOW_CORE_JOB_SCOPE_H_

#include "oneflow/core/job/scope.pb.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/common/maybe.h"

namespace oneflow {

class OperatorConf;

class Scope final {
 public:
  Scope(const Scope&) = delete;
  Scope(Scope&&) = delete;
  explicit Scope(const ScopeProto& scope_proto);
  ~Scope() = default;

  Maybe<const JobDesc*> job_desc() const;
  Maybe<int64_t> GetParallelDescSymbolId(const OperatorConf& op_conf) const;
  Maybe<const ParallelDesc*> GetParallelDesc(const OperatorConf& op_conf) const;

  const OptMirroredParallel& opt_mirrored_parallel_conf() const {
    return scope_proto_.opt_mirrored_parallel_conf();
  }

 private:
  Maybe<void> Init();

  const ScopeProto scope_proto_;
  std::shared_ptr<JobDesc> job_desc_;
  std::shared_ptr<ParallelDesc> device_parallel_desc_;
  std::shared_ptr<ParallelDesc> host_parallel_desc_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_SCOPE_H_
