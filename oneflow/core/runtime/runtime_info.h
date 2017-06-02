#ifndef ONEFLOW_CORE_RUNTIME_RUNTIME_INFO_H_
#define ONEFLOW_CORE_RUNTIME_RUNTIME_INFO_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/id_manager.h"
#include "oneflow/core/runtime/runtime_state.pb.h"

namespace oneflow {

class RuntimeInfo final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RuntimeInfo);
  ~RuntimeInfo() = default;

  static RuntimeInfo& Singleton() {
    static RuntimeInfo obj;
    return obj;
  }

  uint64_t this_machine_id() const { return this_machine_id_; }
  RuntimeState state() const { return state_; }

  void set_this_machine_name(const std::string& name) {
    this_machine_name_ = name;
    this_machine_id_ = IDMgr::Singleton().MachineID4MachineName(name);
    LOG(INFO) << "this machine name: " << this_machine_name_;
    LOG(INFO) << "this machine id: " << this_machine_id_;
  }

 private:
  RuntimeInfo() = default;

  uint64_t this_machine_id_;
  std::string this_machine_name_;
  RuntimeState state_;

};

} // namespace oneflow

#endif // ONEFLOW_CORE_RUNTIME_RUNTIME_INFO_H_
