#ifndef ONEFLOW_CORE_JOB_RUNTIME_BUFFERS_SCOPE_H_
#define ONEFLOW_CORE_JOB_RUNTIME_BUFFERS_SCOPE_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/plan.pb.h"

namespace oneflow {

class RuntimeBuffersScope final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RuntimeBuffersScope);
  RuntimeBuffersScope(const Plan& plan);
  ~RuntimeBuffersScope();
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_RUNTIME_BUFFERS_SCOPE_H_
