#ifndef ONEFLOW_CORE_JOB_RUNTIME_BUFFERS_SCOPE_H_
#define ONEFLOW_CORE_JOB_RUNTIME_BUFFERS_SCOPE_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

class RuntimeBuffersScope final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RuntimeBuffersScope);
  RuntimeBuffersScope();
  ~RuntimeBuffersScope();
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_RUNTIME_BUFFERS_SCOPE_H_
