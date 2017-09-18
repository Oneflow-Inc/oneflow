#ifndef ONEFLOW_CORE_SCHEDULE_EMPTY_UTILIZATION_ANALYZER_H_
#define ONEFLOW_CORE_SCHEDULE_EMPTY_UTILIZATION_ANALYZER_H_

#include "oneflow/core/schedule/utilization_analyzer.h"

namespace oneflow {
namespace schedule {

class EmptyUtilizationAnalyzer : public UtilizationAnalyzer {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EmptyUtilizationAnalyzer);
  EmptyUtilizationAnalyzer() : UtilizationAnalyzer() {}
  ~EmptyUtilizationAnalyzer() = default;

 protected:
  std::unique_ptr<UtilizationGraph> Analyze(
      const SGraph& sgraph,
      const UtilizationEventPackageProto& event_package) const override {
    return of_make_unique<UtilizationGraph>(sgraph);
  }
};

}  // namespace schedule
}  // namespace oneflow

#endif  // ONEFLOW_CORE_SCHEDULE_EMPTY_UTILIZATION_ANALYZER_H_
