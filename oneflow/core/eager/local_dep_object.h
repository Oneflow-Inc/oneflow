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
#ifndef ONEFLOW_CORE_FRAMEWORK_LOCAL_DEP_OBJECT_H_
#define ONEFLOW_CORE_FRAMEWORK_LOCAL_DEP_OBJECT_H_

#include "oneflow/core/intrusive/intrusive.h"
#include "oneflow/core/vm/vm_object.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/common/small_vector.h"
#include "oneflow/core/common/op_args_reserved_size.h"
#include "oneflow/core/framework/device.h"

namespace oneflow {

// LocalDepObject helps VirtualMachineEngine building instruction edges
using LocalDepObject = vm::Dependence;

using DependenceVector = small_vector<LocalDepObject*>;

intrusive::shared_ptr<LocalDepObject> NewLocalDepObject();

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_LOCAL_DEP_OBJECT_H_
