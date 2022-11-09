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
#include "oneflow/core/job/job_ir.h"

namespace oneflow {

#ifndef WITH_MLIR

Maybe<std::string> ConvertJobToTosaIR(Job* job) {
  UNIMPLEMENTED_THEN_RETURN() << "ConvertJobToTosaIR is only supported WITH_MLIR";
}

Maybe<void> SaveJobToIR(Job* job, const std::string& path) {
  UNIMPLEMENTED_THEN_RETURN() << "SaveJobToIR is only supported WITH_MLIR";
}

Maybe<std::string> ConvertJobToIR(Job* job) {
  UNIMPLEMENTED_THEN_RETURN() << "ConvertJobToIR is only supported WITH_MLIR";
}

Maybe<void> LoadJobFromIR(Job* job, const std::string& path) {
  UNIMPLEMENTED_THEN_RETURN() << "LoadJobFromIR is only supported WITH_MLIR";
}

#endif

}  // namespace oneflow
