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
#ifndef ONEFLOW_API_PYTHON_VM_ID_UTIL_H_
#define ONEFLOW_API_PYTHON_VM_ID_UTIL_H_

#include "oneflow/core/common/global.h"
#include "oneflow/core/job/machine_context.h"
#include "oneflow/core/vm/id_util.h"

namespace oneflow {

inline Maybe<long long> NewLogicalObjectId() {
  CHECK_OR_RETURN(JUST(GlobalMaybe<MachineCtx>())->IsThisMachineMaster());
  return vm::IdUtil::NewLogicalObjectId();
}

inline Maybe<long long> NewLogicalSymbolId() {
  CHECK_OR_RETURN(JUST(GlobalMaybe<MachineCtx>())->IsThisMachineMaster());
  return vm::IdUtil::NewLogicalSymbolId();
}

inline Maybe<long long> NewPhysicalObjectId() {
  CHECK_NOTNULL_OR_RETURN(Global<MachineCtx>::Get());
  return vm::IdUtil::NewPhysicalObjectId(Global<MachineCtx>::Get()->this_machine_id());
}

inline Maybe<long long> NewPhysicalSymbolId() {
  CHECK_NOTNULL_OR_RETURN(Global<MachineCtx>::Get());
  return vm::IdUtil::NewPhysicalSymbolId(Global<MachineCtx>::Get()->this_machine_id());
}

}  // namespace oneflow

#endif  // ONEFLOW_API_PYTHON_VM_ID_UTIL_H_
