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
#include "oneflow/core/common/optional.h"
#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/framework/op_interpreter.h"
#include "oneflow/core/framework/random_generator.h"
#include "oneflow/core/functional/function_library.h"
#include "oneflow/core/functional/impl/common.h"
#include "oneflow/core/functional/impl/unary_functor.h"
#include "oneflow/user/kernels/bernoulli_kernel.h"

namespace oneflow {
namespace one {
namespace functional {

namespace impl {

class BernoulliFunctor {
 public:
  BernoulliFunctor() {
    bernoulli_op_ = CHECK_JUST(one::OpBuilder("bernoulli").Input("in").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const DataType& dtype,
                           const Optional<one::Generator>& generator) const {
    MutableAttrMap bernoulli_attrs;
    JUST(bernoulli_attrs.SetAttr<DataType>("dtype", dtype));

    std::shared_ptr<one::Generator> gen;
    if (!generator) {
      gen = JUST(one::DefaultAutoGenerator());
    } else {
      gen = JUST(generator.value());
    }

    JUST(bernoulli_attrs.SetAttr<int64_t>("seed", gen->current_seed()));

    const auto& bernoulli_kernel_state = std::make_shared<BernoulliKernelState>(gen);

    return OpInterpUtil::Dispatch<Tensor>(
        *bernoulli_op_, {x}, OpExprInterpContext(bernoulli_attrs, bernoulli_kernel_state));
  }

 private:
  std::shared_ptr<OpExpr> bernoulli_op_;
};

}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) { m.add_functor<impl::BernoulliFunctor>("Bernoulli"); };

}  // namespace functional
}  // namespace one
}  // namespace oneflow
