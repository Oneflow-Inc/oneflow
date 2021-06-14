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

class NormalizationFunctor {
 public:
  NormalizationFunctor() {
    norm_eval_op_ = CHECK_JUST(one::OpBuilder("normalization")
                                   .Input("x")
                                   .Input("moving_mean")
                                   .Input("moving_variance")
                                   .Input("gamma")
                                   .Input("beta")
                                   .Output("y")
                                   .Attr("training", false)
                                   .Build());
    norm_training_op_ = CHECK_JUST(one::OpBuilder("normalization")
                                       .Input("x")
                                       .Input("moving_mean")
                                       .Input("moving_variance")
                                       .Input("gamma")
                                       .Input("beta")
                                       .Output("y")
                                       .Output("mean")
                                       .Output("inv_variance")
                                       .Attr("training", true)
                                       .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& moving_mean,
                           const std::shared_ptr<one::Tensor>& moving_variance,
                           const std::shared_ptr<one::Tensor>& gamma,
                           const std::shared_ptr<one::Tensor>& beta, const int32_t& axis,
                           const float& epsilon, const float& momentum,
                           const bool& is_training) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("axis", axis));
    JUST(attrs.SetAttr<float>("epsilon", epsilon));
    JUST(attrs.SetAttr<float>("momentum", momentum));
    std::shared_ptr<OpExpr> op;
    if (is_training) {
      op = norm_training_op_;
    } else {
      op = norm_eval_op_;
    }
    return OpInterpUtil::Dispatch<one::Tensor>(*op, {x, moving_mean, moving_variance, gamma, beta},
                                               attrs);
  }

 private:
  std::shared_ptr<OpExpr> norm_eval_op_;
  std::shared_ptr<OpExpr> norm_training_op_;
};

}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) { m.add_functor<impl::NormalizationFunctor>("Normalization"); };

}  // namespace functional
}  // namespace one
}  // namespace oneflow
