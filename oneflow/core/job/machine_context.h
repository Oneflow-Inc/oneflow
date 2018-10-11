#ifndef ONEFLOW_CORE_JOB_MACHINE_CONTEXT_H_
#define ONEFLOW_CORE_JOB_MACHINE_CONTEXT_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

class MachineCtx final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MachineCtx);
  ~MachineCtx() = default;

  int64_t this_machine_id() const { return this_machine_id_; }
  bool IsThisMachineMaster() const { return this_machine_id_ == 0; }

  std::string GetThisAddr() const { return GetAddr(this_machine_id_); }
  std::string GetThisCtrlServerPort() const { return GetCtrlServerPort(this_machine_id_); }
  std::string GetThisCtrlClientPort() const { return GetCtrlClientPort(this_machine_id_); }
  std::string GetAddr(int64_t machine_id) const;
  std::string GetCtrlServerPort(int64_t machine_id) const;
  std::string GetCtrlClientPort(int64_t machine_id) const;

 private:
  friend class Global<MachineCtx>;
  MachineCtx();

  int64_t this_machine_id_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_MACHINE_CONTEXT_H_
