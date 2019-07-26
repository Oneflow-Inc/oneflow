#ifndef ONEFLOW_CORE_JOB_RUNTIME_BUFFER_MANAGERS_SCOPE_H_
#define ONEFLOW_CORE_JOB_RUNTIME_BUFFER_MANAGERS_SCOPE_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

class RuntimeBufferManagersScope final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RuntimeBufferManagersScope);
  RuntimeBufferManagersScope();
  ~RuntimeBufferManagersScope();
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_RUNTIME_BUFFER_MANAGERS_SCOPE_H_
