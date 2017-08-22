#ifndef ONEFLOW_CORE_SCHEDULE_SESSION_FACTORY_H_
#define ONEFLOW_CORE_SCHEDULE_SESSION_FACTORY_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/schedule/factory_util.h"
#include "oneflow/core/schedule/sgraph.h"

namespace oneflow {
namespace schedule {

class ScheduleFactoryProvider;

class SessionFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SessionFactory);
  SessionFactory() = default;
  explicit SessionFactory(ScheduleFactoryProvider*){};
  virtual ~SessionFactory() = default;
  DEFINE_FACTORY_METHOD_CLONE(SessionFactory, SessionFactory);

  virtual std::unique_ptr<Session> CreateSession(const SGraph& graph) const {
    return of_make_unique<Session>(const_cast<SGraph*>(&graph));
  }
};

template<typename SessionType>
class SessionConcreteFactory : public SessionFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SessionConcreteFactory);
  SessionConcreteFactory() = default;
  explicit SessionConcreteFactory(ScheduleFactoryProvider*){};
  virtual ~SessionConcreteFactory() = default;
  std::unique_ptr<Session> CreateSession(const SGraph& graph) const {
    return of_make_unique<SessionType>(const_cast<SGraph*>(&graph));
  }
};

}  // namespace schedule
}  // namespace oneflow
#endif  // ONEFLOW_CORE_SCHEDULE_SESSION_FACTORY_H_
