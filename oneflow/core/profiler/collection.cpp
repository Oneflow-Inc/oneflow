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
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include "nlohmann/json.hpp"
#include "oneflow/core/profiler/collection.h"
#include "oneflow/core/profiler/util.h"

using json = nlohmann::json;

namespace nlohmann {

void to_json(json& j, const std::shared_ptr<::oneflow::profiler::IEvent>& event) {
  j = event->ToJson();
}

}  // namespace nlohmann

namespace oneflow {

namespace profiler {

nlohmann::json IEvent::ToJson() {
  return json{{"name", name_},
              {"time", static_cast<double>(GetDuration())
                           / 1000},  // convert to us,the unit of GetDuration is ns
              {"input_shapes", "-"},
              {"on_gpu", false}};
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
    j["time"] = time_in_us;
    j["on_gpu"] = true;
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
  if (!cuda_event_pair_) {
    IEvent::Start();
  } else {
    cuda_event_pair_->Start();
  }
#else
  IEvent::Start();
#endif
}

void KernelEvent::Finish() {
#if defined(WITH_CUDA)
  if (!cuda_event_pair_) {
    IEvent::Finish();
  } else {
    cuda_event_pair_->Finish();
  }
#else
  IEvent::Finish();
#endif  // WITH_CUDA
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

std::string ProfileMgr::RegisterEventRecorder(const std::shared_ptr<EventRecorder>& event_recorder,
                                              const std::string& name) {
  std::string recorder_key = GetNextEventRecorderKey(name);
  event_recorders_.emplace(recorder_key, event_recorder);
  return recorder_key;
}
void ProfileMgr::UnregisterEventRecorder(const std::string& event_recorder_key) {
  if (event_recorders_.find(event_recorder_key) != event_recorders_.end()) {
    event_recorders_.erase(event_recorder_key);
  }
}

std::string ProfileMgr::DumpResultsJson() {
  const json j = ExportEvents();
  return j.dump();
}

std::vector<std::shared_ptr<IEvent>> ProfileMgr::ExportEvents() {
  std::vector<std::shared_ptr<IEvent>> events;
  while (!events_.empty()) {
    auto e = events_.front();
    events_.pop();
    events.emplace_back(e);
  }
  return events;
}

std::string ProfileMgr::GetNextEventRecorderKey(const std::string& name) {
  if (event_recorders_last_id_.find(name) == event_recorders_last_id_.end()) {
    event_recorders_last_id_[name] = 0;
  } else {
    event_recorders_last_id_[name]++;
  }
  return name + "." + std::to_string(event_recorders_last_id_[name]);
}

std::shared_ptr<EventRecorder> EventRecorder::CreateCustomEventRecorder(const std::string& name) {
  return std::make_shared<EventRecorder>(CustomEvent::Create(name));
}

Maybe<EventRecorder> EventRecorder::CreateKernelEventRecorder(
    const std::string& name,
#if defined(WITH_CUDA)
    cudaStream_t cuda_stream, const std::function<int64_t()>& memory_size_getter,
#endif
    const ShapeGetterFuncType& shape_getter) {
  auto pmgr = Global<ProfileMgr>::Get();
  if (pmgr) {
#if defined(WITH_CUDA)
    if ((pmgr->use_cpu_ && (!cuda_stream)) || (pmgr->use_cuda_ && cuda_stream)) {
      auto event = KernelEvent::Create(name, pmgr->record_shapes_ ? shape_getter : nullptr);
      if (pmgr->use_cuda_ && cuda_stream) {
        event->InitCudaEventPair(cuda_stream);
        if (pmgr->record_bandwidth_) { event->SetMemorySize(memory_size_getter()); }
      }
      return std::make_shared<EventRecorder>(event);
    }
#else  // WITH_CUDA
    if (pmgr->use_cpu_) {
      return std::make_shared<EventRecorder>(
          KernelEvent::Create(name, pmgr->record_shapes_ ? shape_getter : nullptr));
    }
#endif  // WITH_CUDA
  }

  std::shared_ptr<EventRecorder> null_recorder;
  return null_recorder;
}

}  // namespace profiler
}  // namespace oneflow
