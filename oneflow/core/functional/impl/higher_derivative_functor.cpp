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

#include <functional>
#include "oneflow/core/common/scalar.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/functional/functional_api.yaml.h"
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

class SiluGradGradFunctor {
 public:
  // y     = x ∗ sigmoid(x)
  // y'    = (sig(x) + x * sig_grad(x))
  // y''   = (sig(x) + x*sig_grad(x))' = sig_grad(x)*(x+2-2*silu(x))
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x,
                           const std::shared_ptr<Tensor>& dydx) const {
    auto res = functional::sequence_function(functional::Silu)
                   .then([](const std::shared_ptr<Tensor>& input) {
                     return functional::ScalarSub(Scalar(2.0), input, /*alpha=*/Scalar(2.0));
                   })
                   .then([&x](const std::shared_ptr<Tensor>& input) {
                     return functional::Add(x, input, /*alpha=*/Scalar(1.0), /*inplace=*/false);
                   })
                   // Since we use y to compute SigmoidGrad, here we need to use sigmoid with x to
                   // compute x first.
                   // TODO(zzk):  Implement SigmoidGradXDy func.
                   .then(std::bind(functional::SigmoidGrad, JUST(functional::Sigmoid(x)),
                                   std::placeholders::_1))
                   .then(std::bind(functional::Mul, dydx, std::placeholders::_1))
                   .call(x);
    return res;
  }
};

class SeluGradGradFunctor {
 public:
  // y'' = scale * alpha * exp(x) (x < 0)
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x,
                           const std::shared_ptr<Tensor>& dydx) const {
    auto condition = JUST(functional::ScalarLogicalLess(x, Scalar(0.0)));
    auto res = functional::Where(condition, JUST(functional::SeluGrad(dydx, x)),
                                 JUST(functional::ZerosLike(x)));
    return res;
  }
};

class SoftSignGradGradFunctor {
 public:
  // y = x/(1+abs(x)), y' = 1/(1+abs(x))^2, y'' = -2/(1+abs(x))^3*abs_grad(x)
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x,
                           const std::shared_ptr<Tensor>& dydx) const {
    auto res = functional::sequence_function(functional::Abs)
                   .then([](const std::shared_ptr<Tensor>& input) {
                     return functional::ScalarAdd(Scalar(1.0), input, /*alpha=*/Scalar(1));
                   })
                   .then([](const std::shared_ptr<Tensor>& input) {
                     return functional::ScalarPow(input, Scalar(-3), /*inplace=*/false);
                   })
                   .then([](const std::shared_ptr<Tensor>& input) {
                     return functional::ScalarMul(Scalar(-2), input);
                   })
                   .then(std::bind(functional::AbsGrad, x, std::placeholders::_1))
                   .then(std::bind(functional::Mul, dydx, std::placeholders::_1))
                   .call(x);
    return res;
  }
};

class HardSigmoidGradGradFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x,
                           const std::shared_ptr<Tensor>& dydx) const {
    return functional::ZerosLike(x);
  }
};

class HardSwishGradGradFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x,
                           const std::shared_ptr<Tensor>& dydx) const {
    auto condition = JUST(functional::ScalarLogicalGreater(
        (JUST(functional::ScalarLogicalLess(x, Scalar(3.0)))), Scalar(-3.0)));
    return functional::Where(condition, JUST(functional::ScalarDiv(dydx, Scalar(3.0))),
                             JUST(functional::ZerosLike(x)));
  }
};

class SoftplusGradGradFunctor {
 public:
  // beta*x <= threshold:
  // y = 1/beta*ln(1+exp(beta*x)), y' = 1/(1+exp(beta*x))*exp(beta*x)
  // y'' = beta*exp(beta*x)/(1+exp(beta*x))^2 = beta*sig(beta*x)(1-sig(beta*x))
  //     = beta*sig_grad(beta*x)
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x, const std::shared_ptr<Tensor>& dydx,
                           const double& beta, const double& threshold) const {
    auto beta_x = JUST(functional::ScalarMul(x, beta, /*inplace=*/false));
    auto condition = JUST(functional::ScalarLogicalLess(beta_x, Scalar(threshold)));
    auto zero_out = JUST(functional::ZerosLike(x));
    auto res = functional::sequence_function(functional::Sigmoid)
                   .then(std::bind(functional::SigmoidGrad, std::placeholders::_1, dydx))
                   .then([&beta](const std::shared_ptr<Tensor>& input) {
                     return functional::ScalarMul(Scalar(beta), input);
                   })
                   .then(std::bind(functional::Where, condition, std::placeholders::_1, zero_out))
                   .call(beta_x);

    return res;
  }
};

class EluGradGradFunctor {
 public:
  // y = max(0,x) + min(0,alpha∗(exp(x)−1))
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x, const std::shared_ptr<Tensor>& dydx,
                           const double& alpha) const {
    auto condition = JUST(functional::ScalarLogicalLess(x, Scalar(0.0)));
    return functional::Where(condition, JUST(functional::EluGrad(x, dydx, alpha)),
                             JUST(functional::ZerosLike(x)));
  }
};

class CeluGradGradFunctor {
 public:
  // y = max(0,x) + min(0,alpha∗(exp(x/alpha)−1))
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x, const std::shared_ptr<Tensor>& dydx,
                           const double& alpha) const {
    auto condition = JUST(functional::ScalarLogicalLess(x, Scalar(0)));
    auto a = JUST(functional::CeluGrad(x, dydx, alpha));
    auto b = JUST(functional::ScalarDiv(a, Scalar(alpha)));
    auto r = functional::Where(condition, b, JUST(functional::ZerosLike(x)));
    return r;
  }
};

}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<impl::SinGradGradFunctor>("SinGradGrad");
  m.add_functor<impl::CosGradGradFunctor>("CosGradGrad");
  m.add_functor<impl::SiluGradGradFunctor>("SiluGradGrad");
  m.add_functor<impl::SeluGradGradFunctor>("SeluGradGrad");
  m.add_functor<impl::SoftSignGradGradFunctor>("SoftSignGradGrad");
  m.add_functor<impl::HardSigmoidGradGradFunctor>("HardSigmoidGradGrad");
  m.add_functor<impl::HardSwishGradGradFunctor>("HardSwishGradGrad");
  m.add_functor<impl::SoftplusGradGradFunctor>("SoftplusGradGrad");
  m.add_functor<impl::EluGradGradFunctor>("EluGradGrad");
  m.add_functor<impl::CeluGradGradFunctor>("CeluGradGrad");
}

}  // namespace functional
}  // namespace one
}  // namespace oneflow
