#ifndef ONEFLOW_CORE_JOB_COMPLETER_ALL_REDUCE_ADD_PASS_H_
#define ONEFLOW_CORE_JOB_COMPLETER_ALL_REDUCE_ADD_PASS_H_

#include "oneflow/core/job/job.pb.h"

namespace oneflow {

class AllReduceAddPass final {
 public:
  AllReduceAddPass() = default;
  ~AllReduceAddPass() = default;
  void Apply(Job*) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COMPLETER_ALL_REDUCE_ADD_PASS_H_
