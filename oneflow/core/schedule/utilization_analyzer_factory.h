#ifndef ONEFLOW_CORE_SCHEDULE_UTILIZATION_ANALYZER_FACTORY_H_
#define ONEFLOW_CORE_SCHEDULE_UTILIZATION_ANALYZER_FACTORY_H_

#include "oneflow/core/schedule/utilization_analyzer.h"

namespace oneflow {
namespace schedule {

class UtilizationAnalyzerFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UtilizationAnalyzerFactory);
  UtilizationAnalyzerFactory() = default;
  explicit UtilizationAnalyzerFactory(const ScheduleFactoryProvider&) {}
  virtual ~UtilizationAnalyzerFactory() = default;
  DEFINE_FACTORY_METHOD_CLONE(UtilizationAnalyzerFactory,
                              UtilizationAnalyzerFactory);

  virtual std::unique_ptr<UtilizationAnalyzer> CreateUtilizationAnalyzer()
      const {
    return of_make_unique<UtilizationAnalyzer>();
  }
};

template<typename UA>
class UtilizationAnalyzerConcreteFactory : public UtilizationAnalyzerFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UtilizationAnalyzerConcreteFactory);
  UtilizationAnalyzerConcreteFactory() = default;
  explicit UtilizationAnalyzerConcreteFactory(const ScheduleFactoryProvider&) {}
  virtual ~UtilizationAnalyzerConcreteFactory() = default;
  DEFINE_FACTORY_METHOD_CLONE(UtilizationAnalyzerConcreteFactory,
                              UtilizationAnalyzerFactory);

  virtual std::unique_ptr<UtilizationAnalyzer> CreateUtilizationAnalyzer()
      const override {
    return of_make_unique<UA>();
  }
};

}  // namespace schedule
}  // namespace oneflow

#endif  // ONEFLOW_CORE_SCHEDULE_UTILIZATION_ANALYZER_FACTORY_H_
