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
#ifndef ONEFLOW_CORE_BOXING_EAGER_BOXING_INTERPRETER_H_
#define ONEFLOW_CORE_BOXING_EAGER_BOXING_INTERPRETER_H_

#include "oneflow/core/common/symbol.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/boxing/boxing_dividor.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/placed_nd_sbp.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/sbp_parallel.h"
#include "oneflow/core/boxing/boxing_interpreter_status.h"

namespace oneflow {

class EagerBoxingInterpreter {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EagerBoxingInterpreter);
  EagerBoxingInterpreter() = default;
  virtual ~EagerBoxingInterpreter() = default;

  Maybe<one::Tensor> Interpret(const std::shared_ptr<one::Tensor>& input, Symbol<NdSbp> in_nd_sbp,
                               Symbol<NdSbp> out_nd_sbp, Symbol<ParallelDesc> in_parallel_desc,
                               Symbol<ParallelDesc> out_parallel_desc) const;
  virtual Maybe<BoxingInterpreterStatus> boxing_interpreter_status() const = 0;

 protected:
  virtual Maybe<one::Tensor> InterpretImpl(const std::shared_ptr<one::Tensor>& input,
                                           Symbol<NdSbp> in_nd_sbp, Symbol<NdSbp> out_nd_sbp,
                                           Symbol<ParallelDesc> in_parallel_desc,
                                           Symbol<ParallelDesc> out_parallel_desc) const = 0;
};

using BoxingCheckerT = std::function<Maybe<void>(Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out,
                                                 const Shape& logical_shape)>;
using BoxingFunctionT = std::function<Maybe<one::Tensor>(
    const std::shared_ptr<one::Tensor>& input, Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out)>;

Maybe<BoxingFunctionT> GetBoxingFunction(const std::string& method_name, Symbol<PlacedNdSbp> in,
                                         Symbol<PlacedNdSbp> out, const Shape& logical_shape);

void RegisterBoxingFunction(const std::string& method_name, const BoxingCheckerT& Check,
                            const BoxingFunctionT& BoxingFunction);

inline void RegisterBoxingFunction(
    const std::string& method_name,
    const std::pair<BoxingCheckerT, BoxingFunctionT>& CheckAndBoxing) {
  RegisterBoxingFunction(method_name, CheckAndBoxing.first, CheckAndBoxing.second);
}

class NaiveEagerBoxingInterpreter : public EagerBoxingInterpreter {
 public:
  explicit NaiveEagerBoxingInterpreter(
      const std::shared_ptr<BoxingFunctionT>& boxing_function,
      const std::shared_ptr<BoxingInterpreterStatus>& boxing_interpreter_status)
      : boxing_function_(boxing_function), boxing_interpreter_status_(boxing_interpreter_status) {}
  NaiveEagerBoxingInterpreter(const NaiveEagerBoxingInterpreter&) = delete;
  NaiveEagerBoxingInterpreter(NaiveEagerBoxingInterpreter&&) = delete;
  ~NaiveEagerBoxingInterpreter() override = default;

  Maybe<BoxingInterpreterStatus> boxing_interpreter_status() const override {
    return boxing_interpreter_status_;
  }

 private:
  Maybe<one::Tensor> InterpretImpl(const std::shared_ptr<one::Tensor>& input,
                                   Symbol<NdSbp> in_nd_sbp, Symbol<NdSbp> out_nd_sbp,
                                   Symbol<ParallelDesc> in_parallel_desc,
                                   Symbol<ParallelDesc> out_parallel_desc) const override {
    const auto& in_placed_nd_sbp = JUST(PlacedNdSbp::New(in_nd_sbp, in_parallel_desc));
    const auto& out_placed_nd_sbp = JUST(PlacedNdSbp::New(out_nd_sbp, out_parallel_desc));
    return JUST((*boxing_function_)(input, in_placed_nd_sbp, out_placed_nd_sbp));
  }

  const std::shared_ptr<BoxingFunctionT> boxing_function_;
  const std::shared_ptr<BoxingInterpreterStatus> boxing_interpreter_status_;
};

class BoxingExprIf {
 public:
  BoxingExprIf(const BoxingExprIf&) = default;
  BoxingExprIf(BoxingExprIf&&) = default;
  virtual ~BoxingExprIf() = default;

  virtual Maybe<BoxingInterpreterStatus> Check(Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out,
                                               const Shape& logical_shape) const = 0;
  virtual Maybe<BoxingFunctionT> GetBoxingFunction(Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out,
                                                   const Shape& logical_shape) const = 0;

 protected:
  BoxingExprIf() = default;
};

class AtomicBoxingExpr final : public BoxingExprIf {
 public:
  AtomicBoxingExpr(const AtomicBoxingExpr&) = delete;
  AtomicBoxingExpr(AtomicBoxingExpr&&) = delete;
  ~AtomicBoxingExpr() override = default;

  explicit AtomicBoxingExpr(const std::string& boxing_name)
      : BoxingExprIf(), boxing_name_(boxing_name) {}

  Maybe<BoxingInterpreterStatus> Check(Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out,
                                       const Shape& logical_shape) const override;
  Maybe<BoxingFunctionT> GetBoxingFunction(Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out,
                                           const Shape& logical_shape) const override;

 private:
  const std::string boxing_name_;
};

class DivideAndConquerBoxingExpr final : public BoxingExprIf {
 public:
  DivideAndConquerBoxingExpr(const DivideAndConquerBoxingExpr&) = delete;
  DivideAndConquerBoxingExpr(DivideAndConquerBoxingExpr&&) = delete;
  ~DivideAndConquerBoxingExpr() override = default;

  explicit DivideAndConquerBoxingExpr(const std::shared_ptr<BoxingDividor>& boxing_dividor,
                                      const std::shared_ptr<BoxingExprIf>& lhs_conquer,
                                      const std::shared_ptr<BoxingExprIf>& rhs_conquer)
      : BoxingExprIf(),
        boxing_dividor_(boxing_dividor),
        lhs_conquer_(lhs_conquer),
        rhs_conquer_(rhs_conquer) {}

  Maybe<BoxingInterpreterStatus> Check(Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out,
                                       const Shape& logical_shape) const override;
  Maybe<BoxingFunctionT> GetBoxingFunction(Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out,
                                           const Shape& logical_shape) const override;

 private:
  const std::shared_ptr<BoxingDividor> boxing_dividor_;
  const std::shared_ptr<BoxingExprIf> lhs_conquer_;
  const std::shared_ptr<BoxingExprIf> rhs_conquer_;
};

class OrBoxingExpr final : public BoxingExprIf {
 public:
  OrBoxingExpr(const OrBoxingExpr&) = delete;
  OrBoxingExpr(OrBoxingExpr&&) = delete;
  ~OrBoxingExpr() override = default;

  explicit OrBoxingExpr(const std::shared_ptr<BoxingExprIf>& lhs_boxing,
                        const std::shared_ptr<BoxingExprIf>& rhs_boxing)
      : BoxingExprIf(), lhs_boxing_(lhs_boxing), rhs_boxing_(rhs_boxing) {}

  Maybe<BoxingInterpreterStatus> Check(Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out,
                                       const Shape& logical_shape) const override;
  Maybe<BoxingFunctionT> GetBoxingFunction(Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out,
                                           const Shape& logical_shape) const override;

 private:
  const std::shared_ptr<BoxingExprIf> lhs_boxing_;
  const std::shared_ptr<BoxingExprIf> rhs_boxing_;
};

Maybe<BoxingExprIf> BoxingExpr(const std::string& boxing_name);
Maybe<BoxingExprIf> BoxingExpr(const std::shared_ptr<BoxingDividor>& boxing_dividor,
                               const std::string& lhs_conquer, const std::string& rhs_conquer);
Maybe<BoxingExprIf> BoxingExpr(const std::shared_ptr<BoxingDividor>& boxing_dividor,
                               const std::shared_ptr<BoxingExprIf>& lhs_conquer,
                               const std::string& rhs_conquer);
Maybe<BoxingExprIf> BoxingExpr(const std::shared_ptr<BoxingDividor>& boxing_dividor,
                               const std::string& lhs_conquer,
                               const std::shared_ptr<BoxingExprIf>& rhs_conquer);
Maybe<BoxingExprIf> BoxingExpr(const std::shared_ptr<BoxingDividor>& boxing_dividor,
                               const std::shared_ptr<BoxingExprIf>& lhs_conquer,
                               const std::shared_ptr<BoxingExprIf>& rhs_conquer);

std::shared_ptr<BoxingExprIf> operator|(const std::shared_ptr<BoxingExprIf>& lhs_boxing,
                                        const std::shared_ptr<BoxingExprIf>& rhs_boxing);

Maybe<BoxingExprIf> OptionalBoxing(const std::string& boxing_mame);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_BOXING_EAGER_BOXING_INTERPRETER_H_
