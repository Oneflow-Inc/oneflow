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
#include "oneflow/core/thread/thread_executor_factory.h"
#include "oneflow/core/thread/thread_executor.h"

namespace oneflow {
namespace thread {

namespace {

template<typename T>
std::shared_ptr<thread::ExecutorBase> CreateExecutor() {
  return std::shared_ptr<thread::ExecutorBase>(std::make_shared<T>());
}

}  // namespace

Maybe<thread::ExecutorBase> ExecutorFactory::Create(ExecutorType type) {
  if (type == ExecutorType::kOf) { return CreateExecutor<thread::OfExecutor>(); }
  const auto format_error_msg = [](const auto& name, const auto& option) {
    return fmt::format("{} is not enabled, you should compile oneflow with "
                       "`-DCPU_THREADING_RUNTIMES={}`",
                       name, option);
  };

  if (type == ExecutorType::kTbb) {
    if (!IsTbbEnabled()) { return Error::RuntimeError() << format_error_msg("OneTBB", "TBB"); }
    return CreateExecutor<thread::TbbExecutor>();
  }
  if (type == ExecutorType::kOmp) {
    if (!IsOmpEnabled()) { return Error::RuntimeError() << format_error_msg("OpenMP", "OMP"); }
    return CreateExecutor<thread::OmpExecutor>();
  }
  return CreateExecutor<thread::SeqExecutor>();
}

Maybe<thread::ExecutorBase> ExecutorFactory::Create(const std::string& type) {
  std::unordered_map<std::string, ExecutorType> types = {
      {"SEQ", ExecutorType::kSeq},
      {"OF", ExecutorType::kOf},
      {"TBB", ExecutorType::kTbb},
      {"OMP", ExecutorType::kOmp},
  };
  if (types.find(type) == types.end()) {
    return Error::RuntimeError() << fmt::format("Not supportted cpu threading runtime: {}", type);
  }
  return Create(types[type]);
}

}  // namespace thread
}  // namespace oneflow