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
#include "oneflow/core/profiler/event.h"
#include "oneflow/core/profiler/util.h"

using json = nlohmann::json;

namespace oneflow {

namespace profiler {
nlohmann::json IEvent::ToJson() {
  return json{{"name", name_},
              {"cpu_time", static_cast<double>(GetDuration())
                               / 1000},  // convert to us,the unit of GetDuration is ns
              {"input_shapes", "-"}};
}

void IEvent::Start() { started_at_ = GetTimeNow(); }

void IEvent::Finish() { finished_at_ = GetTimeNow(); }

const std::string& IEvent::GetName() const { return name_; }

time_t IEvent::GetDuration() { return finished_at_ - started_at_; }

nlohmann::json KernelEvent::ToJson() {
  auto j = IEvent::ToJson();
  j["type"] = EventType::kKernel;
  j["input_shapes"] = FormatShapes();
#if defined(WITH_CUDA)
  if (cuda_event_pair_) {
    double time_in_us = cuda_event_pair_->ElapsedTime();
    j["gpu_time"] = time_in_us;
    if (memory_size_ != -1) {
      j["bandwidth"] =
          memory_size_ / (1024.0 * 1024.0 * 1024.0) / (time_in_us / (1000 * 1000));  // GB/s
    }
  }
#endif
  return j;
}

std::string KernelEvent::Key() { return name_ + "." + FormatShapes(); }

std::string KernelEvent::FormatShapes(size_t max_num_to_format) {
  if (input_shapes_.size() == 0) { return "-"; }
  std::string result("[");
  for (size_t i = 0; i < std::min(input_shapes_.size(), max_num_to_format); ++i) {
    if (i != 0) { result += ", "; }
    const std::string current_shape = input_shapes_[i].ToString();
    if (current_shape == "()") {
      result += "scalar";
    } else {
      result += current_shape;
    }
  }
  if (input_shapes_.size() > max_num_to_format) { result += ", ..."; }
  result += "]";
  return result;
}

void KernelEvent::RecordShape(const Shape& shape) { input_shapes_.emplace_back(shape); }

void KernelEvent::Start() {
#if defined(WITH_CUDA)
  if (cuda_event_pair_) { cuda_event_pair_->Start(); }
#endif
  IEvent::Start();
}

void KernelEvent::Finish() {
#if defined(WITH_CUDA)
  if (cuda_event_pair_) { cuda_event_pair_->Finish(); }
#endif
  IEvent::Finish();
}

std::shared_ptr<KernelEvent> KernelEvent::Create(
    const std::string& name, const std::function<std::vector<Shape>(void)>& shape_getter) {
  return std::shared_ptr<KernelEvent>(new KernelEvent(name, shape_getter));
}

nlohmann::json CustomEvent::ToJson() {
  auto j = IEvent::ToJson();
  j["type"] = EventType::kCustom;
  return j;
}

std::string CustomEvent::Key() { return name_; }

std::shared_ptr<CustomEvent> CustomEvent::Create(const std::string& name) {
  return std::shared_ptr<CustomEvent>(new CustomEvent(name));
}

}  // namespace profiler
}  // namespace oneflow