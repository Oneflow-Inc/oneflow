#ifndef ONEFLOW_CORE_SCHEDULE_VISUALIZATION_H_
#define ONEFLOW_CORE_SCHEDULE_VISUALIZATION_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/schedule/utilization_graph.h"

namespace oneflow {
namespace schedule {

class Visualization {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Visualization);
  Visualization() = default;
  ~Visualization() = default;

  std::string UGraph2DotString(const UtilizationGraph& ugraph) const;
};

}  // namespace schedule
}  // namespace oneflow
#endif  // ONEFLOW_CORE_SCHEDULE_VISUALIZATION_H_
