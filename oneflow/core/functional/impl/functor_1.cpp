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

#include "oneflow/core/functional/impl/functor_0.h"
#include "oneflow/core/functional/function_library.h"

#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/user_op_attr.cfg.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_expr_helper.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"

namespace oneflow {
namespace one {
namespace functional {

namespace impl {

Maybe<Tensor> NormalizationFunctor::operator()(
    const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& moving_mean,
    const std::shared_ptr<one::Tensor>& moving_variance, const std::shared_ptr<one::Tensor>& gamma,
    const std::shared_ptr<one::Tensor>& beta, const int32_t& axis, const float& epsilon,
    const float& momentum, const bool& is_training) const {
  UNIMPLEMENTED_THEN_RETURN();
}

}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) { m.add_functor<impl::NormalizationFunctor>("Normalization"); };

}  // namespace functional
}  // namespace one
}  // namespace oneflow
