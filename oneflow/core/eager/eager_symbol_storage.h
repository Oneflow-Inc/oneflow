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
#ifndef ONEFLOW_CORE_EAGER_EAGER_SYMBOL_STORAGE_H_
#define ONEFLOW_CORE_EAGER_EAGER_SYMBOL_STORAGE_H_

#include "oneflow/core/vm/symbol_storage.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

class Scope;
class ScopeProto;

class JobDesc;
class JobConfigProto;

namespace vm {

template<>
struct ConstructArgType4Symbol<JobDesc> final {
  using type = JobConfigProto;
};

template<>
struct ConstructArgType4Symbol<Scope> final {
  using type = ScopeProto;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_EAGER_SYMBOL_STORAGE_H_
