#ifndef ONEFLOW_CORE_SCHEDULE_SIMULATOR_H_
#define ONEFLOW_CORE_SCHEDULE_SIMULATOR_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/schedule/plan_sgraph.h"
#include "oneflow/core/schedule/utilization.pb.h"

namespace oneflow {
namespace schedule {

class Simulator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Simulator);
  Simulator() = default;
  ~Simulator() = default;

  std::unique_ptr<UtilizationEventPackageProto> Run(const SGraph& sgraph) const;

 private:
};

}  // namespace schedule
}  // namespace oneflow

#endif  // ONEFLOW_CORE_SCHEDULE_SIMULATOR_H_
