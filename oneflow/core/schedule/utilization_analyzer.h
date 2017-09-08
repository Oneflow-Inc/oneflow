#ifndef ONEFLOW_CORE_SCHEDULE_UTILIZATION_ANALYZER_H_
#define ONEFLOW_CORE_SCHEDULE_UTILIZATION_ANALYZER_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/schedule/sgraph.h"
#include "oneflow/core/schedule/utilization.pb.h"
#include "oneflow/core/schedule/utilization_graph.h"

namespace oneflow {
namespace schedule {

class UtilizationAnalyzer {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UtilizationAnalyzer);
  explicit UtilizationAnalyzer(const SGraph* sgraph) : sgraph_(sgraph) {}
  ~UtilizationAnalyzer() = default;

  inline const SGraph* sgraph() const { return sgraph_; }
  virtual void CreateUtilizationFromEvent(
      const UtilizationEventPackageProto& event_package,
      UtilizationPackageProto* utilization_package) const;
  virtual void CreateUtilizationGraph(
      const UtilizationPackageProto& utilization_package,
      UtilizationGraph* ugraph) const;

 private:
  void AddUtilizationProto(
      const std::list<const UtilizationEventProto*>& event_pair,
      UtilizationPackageProto* utilization_package) const;
  const SGraph* sgraph_;
};

}  // namespace schedule
}  // namespace oneflow

#endif  // ONEFLOW_CORE_SCHEDULE_UTILIZATION_ANALYZER_H_
