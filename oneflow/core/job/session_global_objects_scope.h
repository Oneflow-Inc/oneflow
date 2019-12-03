#ifndef ONEFLOW_CORE_JOB_ENVIRONMENT_OBJECTS_SCOPE_H_
#define ONEFLOW_CORE_JOB_ENVIRONMENT_OBJECTS_SCOPE_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/job_set.pb.h"
#include "oneflow/core/common/maybe.h"

namespace oneflow {

class SessionGlobalObjectsScope final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SessionGlobalObjectsScope);
  SessionGlobalObjectsScope();
  ~SessionGlobalObjectsScope();

  Maybe<void> Init(const ConfigProto& config_proto);

 private:
  HashSet<void*> lib_handles_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_ENVIRONMENT_OBJECTS_SCOPE_H_
