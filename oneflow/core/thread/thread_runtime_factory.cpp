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

#include <fmt/core.h>
#include <unordered_map>
#include "oneflow/core/thread/thread_runtime_factory.h"
#include "oneflow/core/thread/thread_runtime.h"

namespace oneflow {
namespace thread {

namespace {

template<typename T>
std::shared_ptr<thread::RuntimeBase> CreateRuntime() {
  return std::shared_ptr<thread::RuntimeBase>(std::make_shared<T>());
}

}  // namespace

Maybe<thread::RuntimeBase> RuntimeFactory::Create(RuntimeType type) {
  if (type == RuntimeType::kOf) { return CreateRuntime<thread::OfRuntime>(); }
  const auto format_error_msg = [](const auto& name, const auto& option) {
    return fmt::format("{} is not enabled, you should compile oneflow with "
                       "`-DCPU_THREADING_RUNTIMES={}`",
                       name, option);
  };

  if (type == RuntimeType::kTbb) {
    if (!IsTbbEnabled()) { return Error::RuntimeError() << format_error_msg("OneTBB", "TBB"); }
#ifdef WITH_TBB
    return CreateRuntime<thread::TbbRuntime>();
#endif
  }
  if (type == RuntimeType::kOmp) {
    if (!IsOmpEnabled()) { return Error::RuntimeError() << format_error_msg("OpenMP", "OMP"); }
#ifdef WITH_OMP
    return CreateRuntime<thread::OmpRuntime>();
#endif
  }
  return CreateRuntime<thread::SeqRuntime>();
}

Maybe<thread::RuntimeBase> RuntimeFactory::Create(const std::string& type) {
  std::unordered_map<std::string, RuntimeType> types{
      {"SEQ", RuntimeType::kSeq},
      {"OF", RuntimeType::kOf},
      {"TBB", RuntimeType::kTbb},
      {"OMP", RuntimeType::kOmp},
  };
  if (types.find(type) == types.end()) {
    return Error::RuntimeError() << fmt::format("Not supportted cpu threading runtime: {}", type);
  }
  return Create(types[type]);
}

}  // namespace thread
}  // namespace oneflow