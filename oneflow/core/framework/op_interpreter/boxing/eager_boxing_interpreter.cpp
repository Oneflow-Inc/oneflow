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
#include "oneflow/core/framework/op_interpreter/boxing/eager_boxing_interpreter.h"
#include "oneflow/core/framework/op_interpreter/boxing/eager_boxing_interpreter_mgr.h"

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
  const auto& tensor =
      JUST(InterpretImpl(input, in_nd_sbp, out_nd_sbp, in_parallel_desc, out_parallel_desc));
  const auto& tensor_nd_sbp = JUST(tensor->nd_sbp());
  const auto& tensor_placement = JUST(tensor->parallel_desc());
  CHECK_OR_RETURN(tensor_nd_sbp == out_nd_sbp) << typeid(*this).name();
  CHECK_OR_RETURN(tensor_placement == out_parallel_desc) << typeid(*this).name();
  return tensor;
}

Maybe<EagerBoxingCall> EagerBoxingCall::New(Symbol<cfg::NdSbp> in_nd_sbp,
                                            Symbol<cfg::NdSbp> out_nd_sbp,
                                            Symbol<ParallelDesc> in_parallel_desc,
                                            Symbol<ParallelDesc> out_parallel_desc) {
  const auto* mgr = Global<EagerBoxingInterpreterManager>::Get();
  const auto& boxing_interpreter = JUST(
      mgr->GetEagerBoxingInterpreter(in_nd_sbp, out_nd_sbp, in_parallel_desc, out_parallel_desc));
  return std::shared_ptr<EagerBoxingCall>(new EagerBoxingCall{
      .boxing_interpreter = boxing_interpreter,
      .in_nd_sbp = in_nd_sbp,
      .out_nd_sbp = out_nd_sbp,
      .in_parallel_desc = in_parallel_desc,
      .out_parallel_desc = out_parallel_desc,
  });
}

Maybe<one::Tensor> EagerBoxingCall::Apply(const std::shared_ptr<one::Tensor>& input) const {
  const auto& input_nd_sbp = JUST(input->nd_sbp());
  const auto& input_parallel_desc = JUST(input->parallel_desc());
  CHECK_OR_RETURN(input_nd_sbp == this->in_nd_sbp);
  CHECK_OR_RETURN(input_parallel_desc == this->in_parallel_desc);
  return this->boxing_interpreter->Interpret(input, this->in_nd_sbp, this->out_nd_sbp,
                                             this->in_parallel_desc, this->out_parallel_desc);
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

}  // namespace

namespace {

Maybe<BoxingFunctionT> RawGetBoxingFunction(const std::string& method_name, Symbol<PlacedNdSbp> in,
                                            Symbol<PlacedNdSbp> out) {
  const auto& Checker = JUST(MapAt(*MutName2BoxingChecker(), method_name));
  JUST(Checker(in, out));
  return JUST(MapAt(*MutName2BoxingFunction(), method_name));
}

}  // namespace

decltype(GetBoxingFunction) GetBoxingFunction =
    DECORATE(&RawGetBoxingFunction, ThreadLocalCopiable);

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

}  // namespace oneflow
