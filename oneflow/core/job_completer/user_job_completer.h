#ifndef ONEFLOW_CORE_JOB_COMPLETER_USER_JOB_COMPLETER_H_
#define ONEFLOW_CORE_JOB_COMPLETER_USER_JOB_COMPLETER_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

class UserJobCompleter final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UserJobCompleter);
  UserJobCompleter() = default;
  ~UserJobCompleter() = default;

  void Complete(Job* job) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COMPLETER_USER_JOB_COMPLETER_H_
