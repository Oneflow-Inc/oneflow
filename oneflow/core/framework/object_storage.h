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
#ifndef ONEFLOW_CORE_FRAMEWORK_OBJECT_STORAGE_H_
#define ONEFLOW_CORE_FRAMEWORK_OBJECT_STORAGE_H_

#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/framework/object.h"

namespace oneflow {

Maybe<bool> HasSharedOpKernelObject4ParallelConfSymbol(
    const std::shared_ptr<ParallelDesc>& parallel_conf_sym);

Maybe<compatible_py::Object> GetOpKernelObject4ParallelConfSymbol(
    const std::shared_ptr<ParallelDesc>& parallel_conf_sym);

Maybe<void> SetSharedOpKernelObject4ParallelConfSymbol(
    const std::shared_ptr<ParallelDesc>& parallel_conf_sym,
    const std::shared_ptr<compatible_py::Object>& shared_opkernel_object);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_OBJECT_STORAGE_H_
