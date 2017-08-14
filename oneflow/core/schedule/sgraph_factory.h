#ifndef ONEFLOW_CORE_SCHEDULE_SGRAPH_FACTORY_H_
#define ONEFLOW_CORE_SCHEDULE_SGRAPH_FACTORY_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/schedule/factory_util.h"
#include "oneflow/core/schedule/sgraph.h"
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

  virtual std::unique_ptr<SGraph> CreateSGraph(const Plan& plan) const = 0;
  virtual std::unique_ptr<SGraph> CreateSGraph(
      const std::string& name) const = 0;
};

template<typename SGraphType>
class SGraphConcreteFactory : public SGraphFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SGraphConcreteFactory);
  SGraphConcreteFactory() = default;
  explicit SGraphConcreteFactory(ScheduleFactoryProvider*){};
  virtual ~SGraphConcreteFactory() = default;
  DEFINE_FACTORY_METHOD_CLONE(SGraphConcreteFactory, SGraphFactory);

  std::unique_ptr<SGraph> CreateSGraph(const Plan& plan) const {
    return unique_ptr_new<SGraphType>(plan);
  }

  std::unique_ptr<SGraph> CreateSGraph(const std::string& name) const {
    return unique_ptr_new<SGraphType>(name);
  }
};

}  // namespace schedule
}  // namespace oneflow
#endif  // ONEFLOW_CORE_SCHEDULE_SGRAPH_FACTORY_H_
