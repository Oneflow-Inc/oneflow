#ifndef ONEFLOW_CORE_PROFILER_EVENT_RECORDER_H_
#define ONEFLOW_CORE_PROFILER_EVENT_RECORDER_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/profiler/event.h"

namespace oneflow {
namespace profiler {

class EventRecorder {
 public:
  using ShapeGetterFuncType = std::function<std::vector<Shape>(void)>;

  OF_DISALLOW_COPY_AND_MOVE(EventRecorder);

  explicit EventRecorder(const std::shared_ptr<IEvent>& event) : event_(event) {
    CHECK_JUST(RegisterEventToProfileManager(event));
    event_->Start();
  }

  Maybe<void> RegisterEventToProfileManager(const std::shared_ptr<IEvent>& event);

  ~EventRecorder() {
    if (event_) {
      event_->Finish();
      event_.reset();
    }
  }
  static std::shared_ptr<EventRecorder> CreateCustomEventRecorder(const std::string& name);

  static Maybe<EventRecorder> CreateKernelEventRecorder(
      const std::string& name,
#if defined(WITH_CUDA)
      const std::function<int64_t()>& memory_size_getter,
#endif
      const ShapeGetterFuncType& shape_getter);

 private:
  std::shared_ptr<IEvent> event_;
};

}  // namespace profiler
}  // namespace oneflow

#endif  // ONEFLOW_CORE_PROFILER_EVENT_RECORDER_H_
