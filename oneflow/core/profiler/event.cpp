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

#include "fmt/core.h"
#include "fmt/format.h"
#include "oneflow/core/profiler/event.h"
#include "oneflow/core/profiler/util.h"

using json = nlohmann::json;

namespace oneflow {

namespace profiler {
nlohmann::json IEvent::ToJson() {
  return json{{"name", name_}, {"time", GetDuration<double>()}, {"input_shapes", "-"}};
}

void IEvent::SetStartedAt(double t) { started_at_ = t; }

void IEvent::SetFinishedAt(double t) { finished_at_ = t; }

void IEvent::Start() { SetStartedAt(GetTimeNow()); }

void IEvent::Finish() { SetFinishedAt(GetTimeNow()); }

bool IEvent::IsChildOf(const IEvent* e) {
  if (!e) { return false; }
  if (this == e) { return false; }
  return GetStartedAt<double>() >= e->GetStartedAt<double>()
         && GetFinishedAt<double>() <= e->GetFinishedAt<double>();
}

const std::string& IEvent::GetName() const { return name_; }

std::string CustomEvent::Key() { return name_; }

nlohmann::json CustomEvent::ToJson() {
  auto j = IEvent::ToJson();
  j["type"] = EventType::kCustom;
  j["custom_type"] = type_;
  return j;
}

std::shared_ptr<CustomEvent> CustomEvent::Create(const std::string& name, CustomEventType type) {
  return std::shared_ptr<CustomEvent>(new CustomEvent(name, type));
}

std::string KernelEvent::Key() { return fmt::format("{}.{}", name_, GetFormatedInputShapes()); }

nlohmann::json KernelEvent::ToJson() {
  auto j = IEvent::ToJson();
  j["type"] = EventType::kOneflowKernel;
  j["input_shapes"] = GetFormatedInputShapes();
#if defined(WITH_CUDA)
  j["memory_size"] = memory_size_;
  if (!children_.empty()) { j["children"] = children_; }
#endif  // WITH_CUDA
  return j;
}

std::shared_ptr<KernelEvent> KernelEvent::Create(
    const std::string& name, const std::function<std::vector<ShapeView>(void)>& shape_getter) {
  return std::shared_ptr<KernelEvent>(new KernelEvent(name, shape_getter));
}

void KernelEvent::RecordShape(const ShapeView& shape) { input_shapes_.emplace_back(shape); }

std::string KernelEvent::GetFormatedInputShapes(size_t max_num_to_format) {
  if (input_shapes_.size() == 0) { return "-"; }
  std::vector<std::string> shapes_formated(std::min(input_shapes_.size(), max_num_to_format));
  for (auto i = 0; i < shapes_formated.size(); ++i) {
    const std::string current_shape = input_shapes_[i].ToString();
    shapes_formated[i] = current_shape == "()" ? "scalar" : current_shape;
  }
  if (input_shapes_.size() > max_num_to_format) { shapes_formated.emplace_back("..."); }
  return fmt::format("[{}]", fmt::join(shapes_formated, ", "));
}

}  // namespace profiler
}  // namespace oneflow