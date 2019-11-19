#ifndef ONEFLOW_CORE_JOB_ENVIRONMENT_OBJECTS_SCOPE_H_
#define ONEFLOW_CORE_JOB_ENVIRONMENT_OBJECTS_SCOPE_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/job_set.pb.h"
#include "oneflow/core/common/maybe.h"

namespace oneflow {

class EnvironmentObjectsScope final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EnvironmentObjectsScope);
  EnvironmentObjectsScope();
  ~EnvironmentObjectsScope();

  Maybe<void> Init(const ConfigProto& config_proto);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_ENVIRONMENT_OBJECTS_SCOPE_H_
