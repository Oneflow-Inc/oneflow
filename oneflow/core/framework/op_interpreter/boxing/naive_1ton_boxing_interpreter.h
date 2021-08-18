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
#ifndef ONEFLOW_CORE_FRAMEWORK_OP_INTERPRETER_BOXING_NAIVE_12N_BOXING_INTERPRETER_H_
#define ONEFLOW_CORE_FRAMEWORK_OP_INTERPRETER_BOXING_NAIVE_12N_BOXING_INTERPRETER_H_

#include "oneflow/core/framework/op_interpreter/boxing/eager_boxing_interpreter.h"

namespace oneflow {

class Nccl1ToBBoxingInterpreter final : public EagerBoxingInterpreter {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Nccl1ToBBoxingInterpreter);
  Nccl1ToBBoxingInterpreter() = default;
  ~Nccl1ToBBoxingInterpreter() override = default;

 private:
  Maybe<one::Tensor> InterpretImpl(const std::shared_ptr<one::Tensor>& input,
                                   Symbol<cfg::NdSbp> in_nd_sbp, Symbol<cfg::NdSbp> out_nd_sbp,
                                   Symbol<ParallelDesc> in_parallel_desc,
                                   Symbol<ParallelDesc> out_parallel_desc) const override;
};

class Nccl1ToPBoxingInterpreter final : public EagerBoxingInterpreter {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Nccl1ToPBoxingInterpreter);
  Nccl1ToPBoxingInterpreter() = default;
  ~Nccl1ToPBoxingInterpreter() override = default;

 private:
  Maybe<one::Tensor> InterpretImpl(const std::shared_ptr<one::Tensor>& input,
                                   Symbol<cfg::NdSbp> in_nd_sbp, Symbol<cfg::NdSbp> out_nd_sbp,
                                   Symbol<ParallelDesc> in_parallel_desc,
                                   Symbol<ParallelDesc> out_parallel_desc) const override;
};

class Nccl1ToSBoxingInterpreter final : public EagerBoxingInterpreter {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Nccl1ToSBoxingInterpreter);
  Nccl1ToSBoxingInterpreter() = default;
  ~Nccl1ToSBoxingInterpreter() override = default;

 private:
  Maybe<one::Tensor> InterpretImpl(const std::shared_ptr<one::Tensor>& input,
                                   Symbol<cfg::NdSbp> in_nd_sbp, Symbol<cfg::NdSbp> out_nd_sbp,
                                   Symbol<ParallelDesc> in_parallel_desc,
                                   Symbol<ParallelDesc> out_parallel_desc) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_OP_INTERPRETER_BOXING_NAIVE_12N_BOXING_INTERPRETER_H_
