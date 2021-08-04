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
#include "oneflow/user/kernels/uniform_kernel.h"

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

// - name: "rand"
//   signature: "Tensor Rand(*, Shape shape, DataType dtype, Device device=None, Generator
//   generator=None)" bind_python: True

class RandFunctor {
 public:
  RandFunctor() { op_ = CHECK_JUST(one::OpBuilder("uniform").Output("out").Build()); }
  Maybe<Tensor> operator()(const Shape& shape, const DataType& dtype,
                           const Optional<Symbol<Device>>& device,
                           const Optional<one::Generator>& generator) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<Shape>("shape", shape));
    JUST(attrs.SetAttr<DataType>("dtype", dtype));
    // TODO: deal with datatype
    /*
    tensor([[[0.5450, 0.0850, 0.8039],
             [0.6258, 0.9747, 0.7761]]])
    >>> torch.rand(1,2,3,dtype=torch.doule)

    >>> torch.rand(1,2,3,dtype=torch.double)
    tensor([[[0.2221, 0.6954, 0.5011],
             [0.0680, 0.4570, 0.0813]]], dtype=torch.float64)
    */
    // if (IsIntegralDataType(dtype)) {
    //   JUST(attrs.SetAttr<bool>("is_floating_value", false));
    //   JUST(attrs.SetAttr<int64_t>("integer_value", JUST(value.As<int64_t>())));
    // } else {
    //   JUST(attrs.SetAttr<bool>("is_floating_value", true));
    //   JUST(attrs.SetAttr<double>("floating_value", JUST(value.As<double>())));
    // }
    // {
    //   ParallelDistribution parallel_distribution;
    //   parallel_distribution.mutable_sbp_parallel()->Add()->mutable_broadcast_parallel();
    //   JUST(attrs.SetAttr<std::string>("nd_sbp", PbMessage2TxtString(parallel_distribution)));
    // }

    std::shared_ptr<one::Generator> gen;
    if (!generator) {
      gen = JUST(one::DefaultAutoGenerator());
    } else {
      gen = JUST(generator.value());
    }

    JUST(attrs.SetAttr<int64_t>("seed", gen->current_seed()));

    const auto& bernoulli_kernel_state = std::make_shared<UniformKernelState>(gen);

    if (device.has_value()) {
      Symbol<Device> device_symbol = JUST(device.value());
      return OpInterpUtil::Dispatch<Tensor>(
          *op_, {}, OpExprInterpContext(attrs, device_symbol, bernoulli_kernel_state));
    } else {
      return OpInterpUtil::Dispatch<Tensor>(*op_, {},
                                            OpExprInterpContext(attrs, bernoulli_kernel_state));
    }
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<impl::BernoulliFunctor>("Bernoulli");
  m.add_functor<impl::RandFunctor>("Rand");
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow
