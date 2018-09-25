#ifndef ONEFLOW_CORE_JOB_NCCL_COMM_MANAGER_H_
#define ONEFLOW_CORE_JOB_NCCL_COMM_MANAGER_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/plan.pb.h"

#ifdef WITH_NCCL

#include <nccl.h>

namespace oneflow {

class NcclCommMgr final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclCommMgr);
  ~NcclCommMgr();

  ncclComm_t NcclComm4ActorId(int64_t actor_id) const;

 private:
  friend class Global<NcclCommMgr>;
  NcclCommMgr(const Plan& plan);

  void NcclCommInitRank4Tasks(const std::vector<TaskProto>& tasks, std::vector<ncclComm_t>* comms,
                              ncclUniqueId nccl_unique_id);
  void NcclGetUniqueId4Tasks(const std::vector<TaskProto>& tasks, ncclUniqueId* nccl_unique_id);
  bool IsNcclTaskType(const TaskType& tt) const;
  int32_t GetDeviceId4Task(const TaskProto& task);

  HashMap<int64_t, ncclComm_t> actor_id2comm_;
};

}  // namespace oneflow

#endif  // WITH_NCCL

#endif  // ONEFLOW_CORE_JOB_NCCL_COMM_MANAGER_H_
