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
#ifndef ONEFLOW_CORE_BOXING_EAGER_BOXING_INTERPRETER_MGR_H_
#define ONEFLOW_CORE_BOXING_EAGER_BOXING_INTERPRETER_MGR_H_

#include "oneflow/core/boxing/eager_boxing_interpreter.h"

namespace oneflow {

class EagerBoxingInterpreterManager final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EagerBoxingInterpreterManager);
  EagerBoxingInterpreterManager() = default;
  virtual ~EagerBoxingInterpreterManager() = default;

  Maybe<EagerBoxingInterpreter> GetEagerBoxingInterpreter(Symbol<NdSbp> in_nd_sbp,
                                                          Symbol<NdSbp> out_nd_sbp,
                                                          Symbol<ParallelDesc> in_parallel_desc,
                                                          Symbol<ParallelDesc> out_parallel_desc,
                                                          const Shape& logical_shape) const;
};

template<typename RetT, typename... Args>
struct DisableRecusiveBoxingCall {
  static_assert(is_maybe<RetT>::value, "returned value type must be Maybe<T>.");
  template<RetT (*func)(Args...)>
  static RetT Call(Args... arg) {
    static thread_local bool disable_boxing = false;
    CHECK_OR_RETURN(!disable_boxing);
    disable_boxing = true;
    RetT ret = func(arg...);
    disable_boxing = false;
    return ret;
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_BOXING_EAGER_BOXING_INTERPRETER_MGR_H_
