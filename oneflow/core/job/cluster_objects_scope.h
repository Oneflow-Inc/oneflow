#ifndef ONEFLOW_CORE_JOB_CLUSTER_OBJECTS_SCOPE_H_
#define ONEFLOW_CORE_JOB_CLUSTER_OBJECTS_SCOPE_H_
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/job/cluster_desc.h"

namespace oneflow {

class ClusterObjectsScope final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ClusterObjectsScope);
  ClusterObjectsScope();
  ~ClusterObjectsScope();

  Maybe<void> Init(const ClusterProto& cluster_proto);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_CLUSTER_OBJECTS_SCOPE_H_
