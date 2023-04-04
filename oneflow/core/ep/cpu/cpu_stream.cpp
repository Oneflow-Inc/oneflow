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
#include "oneflow/core/ep/cpu/cpu_stream.h"
#include "oneflow/core/thread/thread_runtime_factory.h"

namespace oneflow {

namespace ep {

DeviceType CpuStream::device_type() const { return DeviceType::kCPU; }

CpuDevice* CpuStream::device() const { return device_; }

Maybe<void> CpuStream::Sync() { return Maybe<void>::Ok(); }

void CpuStream::RecordEvent(Event* /*event*/) {}

Maybe<void> CpuStream::InitThreadRuntime() {
  const auto thread_runtime_type = GetStringFromEnv("OF_THREADING_RUNTIME", [] {
    if (thread::IsTbbEnabled()) { return "TBB"; }
    if (thread::IsOmpEnabled()) { return "OMP"; }
    return "SEQ";
  }());
  thread_runtime_ = JUST(thread::RuntimeFactory::Create(thread_runtime_type));
  return Maybe<void>::Ok();
}

#ifdef WITH_ONEDNN

const std::unique_ptr<ep::OneDnnExecutor>& CpuStream::onednn_executor() const {
  return onednn_executor_;
}

#endif

}  // namespace ep

}  // namespace oneflow
