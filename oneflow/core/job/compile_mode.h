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
#ifndef ONEFLOW_CORE_JOB_COMPILE_MODE_H_
#define ONEFLOW_CORE_JOB_COMPILE_MODE_H_

#include "oneflow/core/common/maybe.h"

namespace oneflow {

enum class CompileMode {
  kInvalid = 0,  // make sure kInvalid is the first CompileMode
  kNaive,
  kRankPerProcess,
  kEnd,  // make sure kEnd is the last CompileMode
};

template<typename DerivedT>
struct CompileModeVisitor {
  template<typename... Args>
  static auto Visit(CompileMode compile_mode, Args&&... args) {
    switch (compile_mode) {
      case CompileMode::kNaive: return DerivedT::VisitNaive(std::forward<Args>(args)...);
      case CompileMode::kRankPerProcess:
        return DerivedT::VisitRankPerProcess(std::forward<Args>(args)...);
      default: {
        LOG(FATAL) << "invalid compile mode";
        return DerivedT::VisitInValid(std::forward<Args>(args)...);
      }
    }
  }
};

Maybe<CompileMode> CurrentCompileMode();

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COMPILE_MODE_H_
