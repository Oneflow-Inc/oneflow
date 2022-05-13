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

#include "oneflow/core/common/scalar.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/functional/sequence_function.h"
#include "oneflow/core/functional/function_library.h"
#include "oneflow/core/functional/impl/common.h"
#include "oneflow/core/functional/impl/unary_functor.h"

namespace oneflow {
namespace one {
namespace functional {

namespace impl {

class SinGradGradFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x,
                           const std::shared_ptr<Tensor>& dydx) const {
    auto res = sequence_function(functional::Sin)
                   .then(functional::Negative)
                   .then(std::bind(functional::Mul, dydx, std::placeholders::_1))
                   .call(x);
    return res;
  }
};

class CosGradGradFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x,
                           const std::shared_ptr<Tensor>& dydx) const {
    auto res = sequence_function(functional::Cos)
                   .then(functional::Negative)
                   .then(std::bind(functional::Mul, dydx, std::placeholders::_1))
                   .call(x);
    return res;
  }
};

}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<impl::SinGradGradFunctor>("SinGradGrad");
  m.add_functor<impl::CosGradGradFunctor>("CosGradGrad");
}

}  // namespace functional
}  // namespace one
}  // namespace oneflow
