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

#include "oneflow/core/functional/impl/binary_functor.h"

#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/functional/function_library.h"
#include "oneflow/core/functional/scalar.h"

namespace oneflow {
namespace one {
namespace functional {

namespace impl {

class FakeQuantizationFunctor {
 public:
  FakeQuantizationFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("fake_quantization")
                         .Input("in")
                         .Input("scale")
                         .Input("zero_point")
                         .Output("out")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& in,
                           const std::shared_ptr<one::Tensor>& scale,
                           const std::shared_ptr<one::Tensor>& zero_point,
                           const std::string quantization_formula, const int32_t& quantization_bit,
                           const std::string quantization_scheme) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<string>("quantization_formula", quantization_formula));
    JUST(attrs.SetAttr<int32_t>("quantization_bit", quantization_bit));
    JUST(attrs.SetAttr<string>("quantization_scheme", quantization_scheme));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {in, scale, zero_point}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) { m.add_functor<impl::FakeQuantizationFunctor>("FakeQuantization"); };

}  // namespace functional
}  // namespace one
}  // namespace oneflow
