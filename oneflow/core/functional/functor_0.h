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

#ifndef ONEFLOW_CORE_FUNCTIONAL_FUNCTOR_0_H_
#define ONEFLOW_CORE_FUNCTIONAL_FUNCTOR_0_H_

#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/functional/scalar.h"

namespace oneflow {
namespace one {
namespace functional {

namespace impl {

class AddFunctor {
 public:
  AddFunctor();
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& y) const;

 private:
  std::shared_ptr<OpExpr> add_op_;
};

class AddNFunctor {
 public:
  AddNFunctor();
  Maybe<Tensor> operator()(const TensorTuple& inputs) const;

 private:
  std::vector<std::shared_ptr<OpExpr>> add_n_op_;
};

class AddScalarFunctor {
 public:
  AddScalarFunctor();
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const Scalar& scalar) const;

 private:
  std::shared_ptr<OpExpr> add_scalar_op_;
};

}  // namespace impl

}  // namespace functional
}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FUNCTIONAL_FUNCTOR_0_H_
