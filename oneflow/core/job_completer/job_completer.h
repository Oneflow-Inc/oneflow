#ifndef ONEFLOW_CORE_JOB_COMPLETER_JOB_COMPLETER_H_
#define ONEFLOW_CORE_JOB_COMPLETER_JOB_COMPLETER_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/graph/op_graph.h"

namespace oneflow {

class JobCompleter final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(JobCompleter);
  JobCompleter() = default;
  ~JobCompleter() = default;

  void Complete(Job* job) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COMPLETER_JOB_COMPLETER_H_
