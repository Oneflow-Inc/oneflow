#ifndef ONEFLOW_CORE_JOB_CLUSTER_H_
#define ONEFLOW_CORE_JOB_CLUSTER_H_

#include "oneflow/core/common/maybe.h"

namespace oneflow {

struct Cluster final {
  static Maybe<void> WorkerLoop();
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_CLUSTER_H_
