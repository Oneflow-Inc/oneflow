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
#include <typeinfo>
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/common/registry_error.h"
#include "oneflow/core/boxing/eager_boxing_interpreter.h"
#include "oneflow/core/framework/tensor_rpc_util.h"
#include "oneflow/core/boxing/eager_boxing_interpreter_mgr.h"

namespace oneflow {

namespace {
Maybe<void> CheckEagerBoxingDataType(DataType val) {
  CHECK_OR_RETURN(val != DataType::kTensorBuffer && val != DataType::kOFRecord)
      << "EagerBoxing only support POD data type.";
  return Maybe<void>::Ok();
}
}  // namespace

Maybe<one::Tensor> EagerBoxingInterpreter::Interpret(const std::shared_ptr<one::Tensor>& input,
                                                     Symbol<cfg::NdSbp> in_nd_sbp,
                                                     Symbol<cfg::NdSbp> out_nd_sbp,
                                                     Symbol<ParallelDesc> in_parallel_desc,
                                                     Symbol<ParallelDesc> out_parallel_desc) const {
  JUST(CheckEagerBoxingDataType(input->dtype()->data_type()));
  DisableCheckConsistentTensorMetaScope disable_meta_check;
  const auto& tensor =
      JUST(InterpretImpl(input, in_nd_sbp, out_nd_sbp, in_parallel_desc, out_parallel_desc));
  const auto& tensor_nd_sbp = JUST(tensor->nd_sbp());
  const auto& tensor_placement = JUST(tensor->parallel_desc());
  CHECK_OR_RETURN(tensor_nd_sbp == out_nd_sbp) << typeid(*this).name();
  CHECK_OR_RETURN(tensor_placement == out_parallel_desc) << typeid(*this).name();
  return tensor;
}

namespace {

HashMap<std::string, BoxingCheckerT>* MutName2BoxingChecker() {
  static HashMap<std::string, BoxingCheckerT> map;
  return &map;
}

HashMap<std::string, BoxingFunctionT>* MutName2BoxingFunction() {
  static HashMap<std::string, BoxingFunctionT> map;
  return &map;
}

Maybe<BoxingFunctionT> RawGetBoxingFunction(const std::string& method_name, Symbol<PlacedNdSbp> in,
                                            Symbol<PlacedNdSbp> out, const Shape& logical_shape) {
  const auto& Checker =
      JUST_MSG(MapAt(*MutName2BoxingChecker(), method_name),
               std::stringstream() << "boxing checker not found. checker_name: " << method_name);
  JUST(Checker(in, out, logical_shape));
  return JUST_MSG(MapAt(*MutName2BoxingFunction(), method_name),
                  std::stringstream()
                      << "boxing function not found. function_name: " << method_name);
}

}  // namespace

Maybe<BoxingFunctionT> GetBoxingFunction(const std::string& method_name, Symbol<PlacedNdSbp> in,
                                         Symbol<PlacedNdSbp> out, const Shape& logical_shape) {
  return DECORATE(&RawGetBoxingFunction, ThreadLocalCopiable)(method_name, in, out, logical_shape);
}

void RegisterBoxingFunction(const std::string& method_name, const BoxingCheckerT& Checker,
                            const BoxingFunctionT& BoxingFunction) {
  CatchRegistryError([&]() -> Maybe<void> {
    CHECK_OR_RETURN(MutName2BoxingChecker()->emplace(method_name, Checker).second)
        << "boxing_method_name: " << method_name;
    CHECK_OR_RETURN(MutName2BoxingFunction()->emplace(method_name, BoxingFunction).second)
        << "boxing_method_name: " << method_name;
    return Maybe<void>::Ok();
  });
}

Maybe<void> AtomicBoxingExpr::Check(Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out,
                                    const Shape& logical_shape) const {
  const auto& Checker =
      JUST_MSG(MapAt(*MutName2BoxingChecker(), boxing_name_),
               std::stringstream() << "boxing checker not found. checker_name: " << boxing_name_);
  JUST(Checker(in, out, logical_shape));
  return Maybe<void>::Ok();
}

Maybe<BoxingFunctionT> AtomicBoxingExpr::GetBoxingFunction(Symbol<PlacedNdSbp> in,
                                                           Symbol<PlacedNdSbp> out,
                                                           const Shape& logical_shape) const {
  return DECORATE(&RawGetBoxingFunction, ThreadLocalCopiable)(boxing_name_, in, out, logical_shape);
}

Maybe<void> DivideAndConquerBoxingExpr::Check(Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out,
                                              const Shape& logical_shape) const {
  const auto& middle = JUST((*boxing_dividor_)(in, out));
  JUST(lhs_conquer_->Check(in, middle, logical_shape));
  JUST(rhs_conquer_->Check(middle, out, logical_shape));
  return Maybe<void>::Ok();
}

Maybe<BoxingFunctionT> DivideAndConquerBoxingExpr::GetBoxingFunction(
    Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out, const Shape& logical_shape) const {
  const auto& middle = JUST((*boxing_dividor_)(in, out));
  const auto& lhs_boxing_func = JUST(lhs_conquer_->GetBoxingFunction(in, middle, logical_shape));
  const auto& rhs_boxing_func = JUST(rhs_conquer_->GetBoxingFunction(middle, out, logical_shape));
  BoxingFunctionT boxing_function = [lhs_boxing_func, rhs_boxing_func, middle, in, out](
                                        const std::shared_ptr<one::Tensor>& tensor,
                                        Symbol<PlacedNdSbp> arg_in,
                                        Symbol<PlacedNdSbp> arg_out) -> Maybe<one::Tensor> {
    CHECK_OR_RETURN(in == arg_in);
    CHECK_OR_RETURN(out == arg_out);
    const auto& middle_tensor = JUST((*lhs_boxing_func)(tensor, in, middle));
    return JUST((*rhs_boxing_func)(middle_tensor, middle, out));
  };
  return boxing_function;
}

Maybe<void> OrBoxingExpr::Check(Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out,
                                const Shape& logical_shape) const {
  if (lhs_boxing_->Check(in, out, logical_shape).IsOk()) { return Maybe<void>::Ok(); }
  return rhs_boxing_->Check(in, out, logical_shape);
}

Maybe<BoxingFunctionT> OrBoxingExpr::GetBoxingFunction(Symbol<PlacedNdSbp> in,
                                                       Symbol<PlacedNdSbp> out,
                                                       const Shape& logical_shape) const {
  if (lhs_boxing_->Check(in, out, logical_shape).IsOk()) {
    return lhs_boxing_->GetBoxingFunction(in, out, logical_shape);
  }
  JUST(rhs_boxing_->Check(in, out, logical_shape));
  return rhs_boxing_->GetBoxingFunction(in, out, logical_shape);
}

Maybe<BoxingExprIf> BoxingExpr(const std::string& boxing_name) {
  JUST(MapAt(*MutName2BoxingChecker(), boxing_name));
  auto boxing_expr = std::make_unique<AtomicBoxingExpr>(boxing_name);
  return std::shared_ptr<BoxingExprIf>(std::move(boxing_expr));
}

Maybe<BoxingExprIf> BoxingExpr(const std::shared_ptr<BoxingDividor>& boxing_dividor,
                               const std::string& lhs_conquer, const std::string& rhs_conquer) {
  return BoxingExpr(boxing_dividor, JUST(BoxingExpr(lhs_conquer)), JUST(BoxingExpr(rhs_conquer)));
}

Maybe<BoxingExprIf> BoxingExpr(const std::shared_ptr<BoxingDividor>& boxing_dividor,
                               const std::shared_ptr<BoxingExprIf>& lhs_conquer,
                               const std::string& rhs_conquer) {
  return BoxingExpr(boxing_dividor, lhs_conquer, JUST(BoxingExpr(rhs_conquer)));
}

Maybe<BoxingExprIf> BoxingExpr(const std::shared_ptr<BoxingDividor>& boxing_dividor,
                               const std::string& lhs_conquer,
                               const std::shared_ptr<BoxingExprIf>& rhs_conquer) {
  return BoxingExpr(boxing_dividor, JUST(BoxingExpr(lhs_conquer)), rhs_conquer);
}

Maybe<BoxingExprIf> BoxingExpr(const std::shared_ptr<BoxingDividor>& boxing_dividor,
                               const std::shared_ptr<BoxingExprIf>& lhs_conquer,
                               const std::shared_ptr<BoxingExprIf>& rhs_conquer) {
  auto divide_and_conquer =
      std::make_unique<DivideAndConquerBoxingExpr>(boxing_dividor, lhs_conquer, rhs_conquer);
  return std::shared_ptr<BoxingExprIf>(std::move(divide_and_conquer));
}

std::shared_ptr<BoxingExprIf> operator|(const std::shared_ptr<BoxingExprIf>& lhs_boxing,
                                        const std::shared_ptr<BoxingExprIf>& rhs_boxing) {
  auto or_boxing = std::make_unique<OrBoxingExpr>(lhs_boxing, rhs_boxing);
  return std::shared_ptr<BoxingExprIf>(std::move(or_boxing));
}

Maybe<BoxingExprIf> OptionalBoxing(const std::string& boxing_mame) {
  return JUST(BoxingExpr(boxing_mame)) | JUST(BoxingExpr("identity"));
}

}  // namespace oneflow
