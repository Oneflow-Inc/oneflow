#ifndef ONEFLOW_CORE_SCHEDULE_SGRAPH_FACTORY_H_
#define ONEFLOW_CORE_SCHEDULE_SGRAPH_FACTORY_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/schedule/factory_util.h"
#include "oneflow/core/schedule/node.h"
#include "oneflow/core/schedule/util.h"

namespace oneflow {
namespace schedule {

class ScheduleFactoryProvider;

class SGraphFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SGraphFactory);
  SGraphFactory() = default;
  virtual ~SGraphFactory() = default;
  DEFINE_FACTORY_PURE_VIRTUAL_CLONE(SGraphFactory);

  virtual std::unique_ptr<SGraph> CreateSGraph(const Plan& plan) = 0;
};

template<typename SGraphType>
class SGraphConcreteFactory : public SGraphFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SGraphConcreteFactory);
  SGraphConcreteFactory() = default;
  explicit SGraphConcreteFactory(ScheduleFactoryProvider*){};
  virtual ~SGraphConcreteFactory() = default;
  DEFINE_FACTORY_METHOD_CLONE(SGraphConcreteFactory, SGraphFactory);

  std::unique_ptr<SGraph> CreateSGraph(const Plan& plan) {
    return SGraphType::CreateFromPlan(plan);
  }
};

}  // namespace schedule
}  // namespace oneflow
#endif  // ONEFLOW_CORE_SCHEDULE_SGRAPH_FACTORY_H_
