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
#ifndef ONEFLOW_CORE_FRAMEWORK_OP_INTERPRETER_BOXING_NAIVE_N_TO_1_BOXING_INTERPRETER_H_
#define ONEFLOW_CORE_FRAMEWORK_OP_INTERPRETER_BOXING_NAIVE_N_TO_1_BOXING_INTERPRETER_H_

#include "oneflow/core/framework/op_interpreter/boxing/eager_boxing_interpreter.h"

namespace oneflow {

class NcclBTo1BoxingInterpreter final : public EagerBoxingInterpreter {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclBTo1BoxingInterpreter);
  NcclBTo1BoxingInterpreter() = default;
  ~NcclBTo1BoxingInterpreter() override = default;

 private:
  Maybe<one::Tensor> InterpretImpl(const std::shared_ptr<one::Tensor>& input,
                                   Symbol<cfg::NdSbp> in_nd_sbp, Symbol<cfg::NdSbp> out_nd_sbp,
                                   Symbol<ParallelDesc> in_parallel_desc,
                                   Symbol<ParallelDesc> out_parallel_desc) const override;
};

class NcclPTo1BoxingInterpreter final : public EagerBoxingInterpreter {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclPTo1BoxingInterpreter);
  NcclPTo1BoxingInterpreter() = default;
  ~NcclPTo1BoxingInterpreter() override = default;

 private:
  Maybe<one::Tensor> InterpretImpl(const std::shared_ptr<one::Tensor>& input,
                                   Symbol<cfg::NdSbp> in_nd_sbp, Symbol<cfg::NdSbp> out_nd_sbp,
                                   Symbol<ParallelDesc> in_parallel_desc,
                                   Symbol<ParallelDesc> out_parallel_desc) const override;
};

class NcclSTo1BoxingInterpreter final : public EagerBoxingInterpreter {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclSTo1BoxingInterpreter);
  NcclSTo1BoxingInterpreter() = default;
  ~NcclSTo1BoxingInterpreter() override = default;

 private:
  Maybe<one::Tensor> InterpretImpl(const std::shared_ptr<one::Tensor>& input,
                                   Symbol<cfg::NdSbp> in_nd_sbp, Symbol<cfg::NdSbp> out_nd_sbp,
                                   Symbol<ParallelDesc> in_parallel_desc,
                                   Symbol<ParallelDesc> out_parallel_desc) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_OP_INTERPRETER_BOXING_NAIVE_N_TO_1_BOXING_INTERPRETER_H_
