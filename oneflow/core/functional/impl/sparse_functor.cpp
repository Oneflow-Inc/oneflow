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
class OneHotFunctor {
 public:
  OneHotFunctor() {
    one_hot_op_ = CHECK_JUST(one::OpBuilder("one_hot").Input("indices").Output("out").Build());
  }
  Maybe<Tensor> operator()(const int64_t& depth, const Scalar& value, const DataType& dtype) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("depth", depth));
    JUST(attrs.SetAttr<DataType>("dtype", dtype));
    if (IsIntegralDataType(dtype)) {
      JUST(attrs.SetAttr<int64_t>("integer_on_value", JUST(value.As<int64_t>())));
      JUST(attrs.SetAttr<int64_t>("integer_off_value", JUST(value.As<int64_t>())));
    } else {
      JUST(attrs.SetAttr<double>("floating_on_value", JUST(value.As<double>())));
      JUST(attrs.SetAttr<double>("floating_off_value", JUST(value.As<double>())));
    }
    return OpInterpUtil::Dispatch<Tensor>(*op_, {}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
}

}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<impl::OneHotFunctor>("OneHot");
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow
