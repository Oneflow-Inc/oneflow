#ifndef ONEFLOW_CORE_SCHEDULE_STATIC_SCHEDULER_H_
#define ONEFLOW_CORE_SCHEDULE_STATIC_SCHEDULER_H_

#include "oneflow/core/schedule/schedule.h"

namespace oneflow {
namespace schedule {

class ScheduleEngine {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ScheduleEngine);
  virtual ~ScheduleEngine() = default;
  explicit ScheduleEngine(const Session* session) : session_(session) {}

  virtual std::unique_ptr<Schedule> StaticSchedule(
      const std::function<uint32_t(uint64_t)>& get_regst_num) = 0;

  std::unique_ptr<Schedule> StaticSchedule(uint32_t regst_max) {
    return StaticSchedule([=](uint64_t id) { return regst_max; });
  }

  virtual std::unique_ptr<Schedule> StaticSchedule() = 0;

  //	getter
  inline const Session* session() const { return session_; }

 protected:
  const Session* session_;
};

}  // namespace schedule
}  // namespace oneflow

#endif  // ONEFLOW_CORE_SCHEDULE_STATIC_SCHEDULER_H_
