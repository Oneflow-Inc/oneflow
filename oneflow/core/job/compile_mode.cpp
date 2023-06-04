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
#include "oneflow/core/job/compile_mode.h"
#include "oneflow/core/common/env_var/env_var.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/container_util.h"

namespace oneflow {

namespace {

struct CompileModeName final : public CompileModeVisitor<CompileModeName> {
  static std::string VisitNaive() { return "naive"; }
  static std::string VisitRankPerProcess() { return "rank_per_process"; }
  static std::string VisitInValid() { return "invalid"; }
};

std::unordered_map<std::string, CompileMode> Name2CompileMode() {
  std::unordered_map<std::string, CompileMode> name2compile_mode;
  for (int i = static_cast<int>(CompileMode::kInvalid) + 1;
       i != static_cast<int>(CompileMode::kEnd); ++i) {
    CompileMode compile_mode = static_cast<CompileMode>(i);
    CHECK(name2compile_mode.emplace(CompileModeName::Visit(compile_mode), compile_mode).second);
  }
  return name2compile_mode;
}

std::string GetValidCompileModeNames() {
  std::stringstream ss;
  for (int i = static_cast<int>(CompileMode::kInvalid) + 1;
       i != static_cast<int>(CompileMode::kEnd); ++i) {
    if (i > static_cast<int>(CompileMode::kInvalid) + 1) { ss << ", "; }
    CompileMode compile_mode = static_cast<CompileMode>(i);
    ss << CompileModeName::Visit(compile_mode);
  }
  return ss.str();
}

}  // namespace

Maybe<CompileMode> CurrentCompileMode() {
  static thread_local CompileMode mode =
      JUST_MSG(MapAt(Name2CompileMode(), ThreadLocalEnvString<ONEFLOW_LAZY_COMPILE_MODE>()),
               std::stringstream()
                   << "ONEFLOW_LAZY_COMPILER(value: "
                   << ThreadLocalEnvString<ONEFLOW_LAZY_COMPILE_MODE>()
                   << ") is invalid. valid options: \"" << GetValidCompileModeNames() << "\"");
  return mode;
}

}  // namespace oneflow
