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
#ifndef ONEFLOW_CORE_FRAMEWORK_OP_INTERPRETER_BOXING_EAGER_BOXING_INTERPRETER_H_
#define ONEFLOW_CORE_FRAMEWORK_OP_INTERPRETER_BOXING_EAGER_BOXING_INTERPRETER_H_

#include "oneflow/core/common/symbol.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/placed_nd_sbp.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/sbp_parallel.cfg.h"

namespace oneflow {

class EagerBoxingInterpreter {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EagerBoxingInterpreter);
  EagerBoxingInterpreter() = default;
  virtual ~EagerBoxingInterpreter() = default;

  Maybe<one::Tensor> Interpret(const std::shared_ptr<one::Tensor>& input,
                               Symbol<cfg::NdSbp> in_nd_sbp, Symbol<cfg::NdSbp> out_nd_sbp,
                               Symbol<ParallelDesc> in_parallel_desc,
                               Symbol<ParallelDesc> out_parallel_desc) const;

 protected:
  virtual Maybe<one::Tensor> InterpretImpl(const std::shared_ptr<one::Tensor>& input,
                                           Symbol<cfg::NdSbp> in_nd_sbp,
                                           Symbol<cfg::NdSbp> out_nd_sbp,
                                           Symbol<ParallelDesc> in_parallel_desc,
                                           Symbol<ParallelDesc> out_parallel_desc) const = 0;
};

struct EagerBoxingCall {
  static Maybe<EagerBoxingCall> New(Symbol<cfg::NdSbp> in_nd_sbp, Symbol<cfg::NdSbp> out_nd_sbp,
                                    Symbol<ParallelDesc> in_parallel_desc,
                                    Symbol<ParallelDesc> out_parallel_desc);

  Maybe<one::Tensor> Apply(const std::shared_ptr<one::Tensor>& input) const;

  const std::shared_ptr<const EagerBoxingInterpreter> boxing_interpreter;
  const Symbol<cfg::NdSbp> in_nd_sbp;
  const Symbol<cfg::NdSbp> out_nd_sbp;
  const Symbol<ParallelDesc> in_parallel_desc;
  const Symbol<ParallelDesc> out_parallel_desc;
};

using BoxingCheckerT = std::function<Maybe<void>(Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out)>;
using BoxingFunctionT = std::function<Maybe<one::Tensor>(
    const std::shared_ptr<one::Tensor>& input, Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out)>;

extern Maybe<BoxingFunctionT> (*GetBoxingFunction)(const std::string& method_name,
                                                   Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out);

void RegisterBoxingFunction(const std::string& method_name, const BoxingCheckerT& Check,
                            const BoxingFunctionT& BoxingFunction);

inline void RegisterBoxingFunction(
    const std::string& method_name,
    const std::pair<BoxingCheckerT, BoxingFunctionT>& CheckAndBoxing) {
  RegisterBoxingFunction(method_name, CheckAndBoxing.first, CheckAndBoxing.second);
}

class NaiveEagerBoxingInterpreter : public EagerBoxingInterpreter {
 public:
  explicit NaiveEagerBoxingInterpreter(const std::shared_ptr<BoxingFunctionT>& boxing_function)
      : boxing_function_(boxing_function) {}
  NaiveEagerBoxingInterpreter(const NaiveEagerBoxingInterpreter&) = delete;
  NaiveEagerBoxingInterpreter(NaiveEagerBoxingInterpreter&&) = delete;
  ~NaiveEagerBoxingInterpreter() override = default;

 private:
  Maybe<one::Tensor> InterpretImpl(const std::shared_ptr<one::Tensor>& input,
                                   Symbol<cfg::NdSbp> in_nd_sbp, Symbol<cfg::NdSbp> out_nd_sbp,
                                   Symbol<ParallelDesc> in_parallel_desc,
                                   Symbol<ParallelDesc> out_parallel_desc) const override {
    const auto& in_placed_nd_sbp = JUST(PlacedNdSbp::New(in_nd_sbp, in_parallel_desc));
    const auto& out_placed_nd_sbp = JUST(PlacedNdSbp::New(out_nd_sbp, out_parallel_desc));
    return JUST((*boxing_function_)(input, in_placed_nd_sbp, out_placed_nd_sbp));
  }

  const std::shared_ptr<BoxingFunctionT> boxing_function_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_OP_INTERPRETER_BOXING_EAGER_BOXING_INTERPRETER_H_
