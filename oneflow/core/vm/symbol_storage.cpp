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
#include "oneflow/core/vm/symbol_storage.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/scope.h"
#include "oneflow/core/operator/op_conf_symbol.h"

namespace oneflow {

namespace symbol {

namespace detail {

template<>
Maybe<ParallelDesc> NewSymbol<ParallelDesc>(
    int64_t symbol_id, const typename ConstructArgType4Symbol<ParallelDesc>::type& data) {
  return ParallelDesc::New(symbol_id, data);
}

template<>
Maybe<JobDesc> NewSymbol<JobDesc>(int64_t symbol_id,
                                  const typename ConstructArgType4Symbol<JobDesc>::type& data) {
  return JobDesc::New(symbol_id, data);
}

template<>
Maybe<Scope> NewSymbol<Scope>(int64_t symbol_id,
                              const typename ConstructArgType4Symbol<Scope>::type& data) {
  return Scope::New(symbol_id, data);
}

template<>
Maybe<OperatorConfSymbol> NewSymbol<OperatorConfSymbol>(
    int64_t symbol_id, const typename ConstructArgType4Symbol<OperatorConfSymbol>::type& data) {
  return std::make_shared<OperatorConfSymbol>(symbol_id, data);
}

}  // namespace detail

}  // namespace symbol

}  // namespace oneflow
