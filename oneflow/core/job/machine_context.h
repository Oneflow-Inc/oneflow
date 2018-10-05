#ifndef ONEFLOW_CORE_JOB_MACHINE_CONTEXT_H_
#define ONEFLOW_CORE_JOB_MACHINE_CONTEXT_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

class MachineCtx final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MachineCtx);
  MachineCtx() = delete;
  ~MachineCtx() = default;

  int64_t this_machine_id() const { return this_machine_id_; }
  bool IsThisMachineMaster() const { return this_machine_id_ == 0; }
  std::string GetMasterCtrlAddr() const { return GetCtrlAddr(0); }
  std::string GetBcastAddr() const { return "0.0.0.0"; }

  std::string GetThisCtrlAddr() const { return GetCtrlAddr(this_machine_id_); }
  std::string GetThisCtrlPort() const { return GetCtrlPort(this_machine_id_); }
  std::string GetThisListenPort() const { return GetListenPort(this_machine_id_); }

  std::string GetCtrlAddr(int64_t machine_id) const;
  std::string GetCtrlPort(int64_t machine_id) const;
  std::string GetListenPort(int64_t machine_id) const;

 private:
  friend class Global<MachineCtx>;
  explicit MachineCtx(int64_t this_mchn_id);

  int64_t this_machine_id_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_MACHINE_CONTEXT_H_
