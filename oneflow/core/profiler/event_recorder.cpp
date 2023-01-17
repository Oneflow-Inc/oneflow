/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/profiler/event_recorder.h"
#include "oneflow/core/profiler/profile_manager.h"
#include "oneflow/core/common/shape_view.h"

namespace oneflow {
namespace profiler {

Maybe<void> EventRecorder::RegisterEventToProfileManager(const std::shared_ptr<IEvent>& event) {
  auto* pmgr = JUST(SingletonMaybe<ProfileManager>());
  pmgr->events_.push(event_);
  return Maybe<void>::Ok();
}

std::shared_ptr<EventRecorder> EventRecorder::CreateCustomEventRecorder(const std::string& name) {
  return std::make_shared<EventRecorder>(CustomEvent::Create(name));
}

Maybe<EventRecorder> EventRecorder::CreateKernelEventRecorder(
    const std::string& name,
#if defined(WITH_CUDA)
    const std::function<int64_t()>& memory_size_getter,
#endif
    const DescriptionGetter& input_shapes_getter, const DescriptionGetter& attrs_getter) {
  auto pmgr = Singleton<ProfileManager>::Get();
  if (pmgr) {
    const auto description_getter = [pmgr, input_shapes_getter, attrs_getter]() {
      KernelEvent::Description desc;
      if (pmgr->record_shapes_) { desc["input_shapes"] = input_shapes_getter(); }
      if (pmgr->record_attrs_) { desc["attrs"] = attrs_getter(); }
      return desc;
    };
#if defined(WITH_CUDA)
    if (pmgr->use_cpu_ || pmgr->use_cuda_) {
      auto event = KernelEvent::Create(name, description_getter());
      if (pmgr->use_cuda_) {
        if (pmgr->record_bandwidth_) { event->SetMemorySize(memory_size_getter()); }
      }
      return std::make_shared<EventRecorder>(event);
    }
#else
    if (pmgr->use_cpu_) {
      return std::make_shared<EventRecorder>(KernelEvent::Create(name, description_getter()));
    }
#endif  // WITH_CUDA
  }

  std::shared_ptr<EventRecorder> null_recorder;
  return null_recorder;
}

}  // namespace profiler
}  // namespace oneflow
