#ifndef	ONEFLOW_CORE_SCHEDULE_UTILIZATION_ANALYZER_FACTORY_H_
#define	ONEFLOW_CORE_SCHEDULE_UTILIZATION_ANALYZER_FACTORY_H_

#include "oneflow/core/schedule/utilization_analyzer.h"

namespace oneflow {
namespace schedule {

class UtilizationAnalyzerFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UtilizationAnalyzerFactory);
  UtilizationAnalyzerFactory() = default;
	explicit UtilizationAnalyzerFactory(ScheduleFactoryProvider*) {}
  virtual ~UtilizationAnalyzerFactory() = default;
  DEFINE_FACTORY_METHOD_CLONE(UtilizationAnalyzerFactory,
			UtilizationAnalyzerFactory);

  virtual std::unique_ptr<UtilizationAnalyzer> CreateUtilizationAnalyzer(
      const SGraph* sgraph) const {
		return of_make_unique<UtilizationAnalyzer>(sgraph);
	}
};

}  // namespace schedule
}  // namespace oneflow

#endif	// ONEFLOW_CORE_SCHEDULE_UTILIZATION_ANALYZER_FACTORY_H_
