#ifndef ONEFLOW_CORE_JOB_NCCL_COMM_MANAGER_H_
#define ONEFLOW_CORE_JOB_NCCL_COMM_MANAGER_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/plan.pb.h"
#include "nccl.h"

namespace oneflow {

class NcclCommMgr final {
 public:
  ~NcclCommMgr();

  ncclComm_t NcclComm4ActorId(int64_t actor_id) const;

 private:
  friend class Global<NcclCommMgr>;
  NcclCommMgr(const Plan& plan);

  bool IsNcclTaskType(const TaskType& tt) const;
  HashMap<int64_t, ncclComm_t> actor_id2comm_;
};
}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_NCCL_COMM_MANAGER_H_
