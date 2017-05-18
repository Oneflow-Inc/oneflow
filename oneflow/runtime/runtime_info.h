#ifndef ONEFLOW_RUNTIME_RUNTIME_INFO_H_
#define ONEFLOW_RUNTIME_RUNTIME_INFO_H_

#include "common/util.h"

namespace oneflow {

enum class RuntimeState {
  kLoadModel
};

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

 private:
  RuntimeInfo() = default;

  uint64_t this_machine_id_;
  RuntimeState state_;

};

} // namespace oneflow

#endif // ONEFLOW_RUNTIME_RUNTIME_INFO_H_
