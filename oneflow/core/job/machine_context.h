#ifndef ONEFLOW_CORE_JOB_MACHINE_CONTEXT_H_
#define ONEFLOW_CORE_JOB_MACHINE_CONTEXT_H_

#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

class MachineCtx final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MachineCtx);
  MachineCtx() = delete;
  ~MachineCtx() = default;

  int64_t this_machine_id() const { return this_machine_id_; }
  bool IsThisMachineMaster() const { return this_machine_id_ == 0; }
  std::string GetThisCtrlAddr() const { return GetCtrlAddr(this_machine_id_); }
  std::string GetMasterCtrlAddr() const { return GetCtrlAddr(0); }
  std::string GetCtrlAddr(int64_t machine_id) const;
#ifdef WITH_CUDA
  ncclUniqueId GetNcclUniqueId() const { return nccl_unique_id_; }
  ncclUniqueId GetNcclScatterUniqueId() const { return nccl_scatter_unique_id_; }
  ncclUniqueId GetNcclGatherUniqueId() const { return nccl_gather_unique_id_; }
#endif

 private:
  friend class Global<MachineCtx>;
  MachineCtx(const std::string& this_mchn_name);
#ifdef WITH_CUDA
  ncclUniqueId nccl_unique_id_;
  ncclUniqueId nccl_scatter_unique_id_;
  ncclUniqueId nccl_gather_unique_id_;
#endif

  int64_t this_machine_id_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_MACHINE_CONTEXT_H_
