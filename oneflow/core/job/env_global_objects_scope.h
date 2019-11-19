#ifndef ONEFLOW_CORE_JOB_CLUSTER_OBJECTS_SCOPE_H_
#define ONEFLOW_CORE_JOB_CLUSTER_OBJECTS_SCOPE_H_
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/job/env_desc.h"

namespace oneflow {

class EnvGlobalObjectsScope final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EnvGlobalObjectsScope);
  EnvGlobalObjectsScope() = default;
  ~EnvGlobalObjectsScope();

  Maybe<void> Init(const EnvProto& env_proto);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_CLUSTER_OBJECTS_SCOPE_H_
