#ifndef ONEFLOW_CORE_JOB_FOREIGN_WATCHER_H_
#define ONEFLOW_CORE_JOB_FOREIGN_WATCHER_H_

#include "oneflow/core/register/ofblob.h"

namespace oneflow {

class ForeignWatcher {
 public:
  ForeignWatcher() = default;
  virtual ~ForeignWatcher() = default;

  virtual void Call(const std::string& handler_uuid, int64_t ofblob_ptr) const { UNIMPLEMENTED(); }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_FOREIGN_WATCHER_H_
