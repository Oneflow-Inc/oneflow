#ifndef ONEFLOW_JOB_RUNTIME_INFO_H_
#define ONEFLOW_JOB_RUNTIME_INFO_H_

#include "common/util.h"

namespace oneflow {

class RuntimeInfo final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RuntimeInfo);
  ~RuntimeInfo() = default;

  static RuntimeInfo& Singleton() {
    static RuntimeInfo obj;
    return obj;
  }

  uint64_t machine_id() const { return machine_id_; }

 private:
  RuntimeInfo() = default;

  uint64_t machine_id_;

};

} // namespace oneflow

#endif // ONEFLOW_JOB_RUNTIME_INFO_H_
