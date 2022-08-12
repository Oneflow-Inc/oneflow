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
  // silu  = x ∗ sigmoid(x)
  // dx    = (sig(x) + x * sig_grad(x))
  // ddx   = (sig(x) + x*sig_grad(x))' = sig_grad(x)*(x+2-2*silu(x))
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x,
                           const std::shared_ptr<Tensor>& dydx) const {
    auto a =
        JUST(functional::ScalarSub(Scalar(2.0), JUST(functional::Silu(x)), /*alpha=*/Scalar(2.0)));
    auto b = JUST(functional::Add(x, a, /*alpha=*/Scalar(1.0), /*inplace=*/true));
    auto c = JUST(functional::SigmoidGrad(x, dydx));
    auto r = JUST(functional::Mul(b, c));
    return r;
  }
};

class MishGradGradFunctor {
 public:
  // Mish(x)=x∗Tanh(Softplus(x))
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x,
                           const std::shared_ptr<Tensor>& dydx) const {
    auto res = x;
    return res;
  }
};

class SeluGradGradFunctor {
 public:
  // ddx = scale * alpha * exp(x) (x < 0)
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x,
                           const std::shared_ptr<Tensor>& dydx) const {
    auto condition = JUST(functional::ScalarLogicalLess(x, Scalar(0.0)));
    return functional::Where(condition, JUST(functional::SeluGrad(x, dydx)),
                             JUST(functional::ZerosLike(x)));
  }
};

class SoftSignGradGradFunctor {
 public:
  // y = x/(1+abs(x)), dx = 1/(1+abs(x))^2, ddx = -2/(1+abs(x))^3*abs_grad(x)
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x,
                           const std::shared_ptr<Tensor>& dydx) const {
    auto a = JUST(functional::ScalarAdd(Scalar(1), JUST(functional::Abs(x)), /*alpha=*/Scalar(1)));
    auto b =
        JUST(functional::ScalarMul(Scalar(-2), JUST(functional::ScalarPow(a, Scalar(-3), false))));
    auto c = JUST(functional::AbsGrad(x, dydx));
    auto r = functional::Mul(b, c);
    return r;
  }
};

class GeluGradGradFunctor {
 public:
  // GELU(x)=x∗gaussian(x)
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x,
                           const std::shared_ptr<Tensor>& dydx) const {
    auto res = x;
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
    auto a = JUST(functional::ScalarDiv(dydx, Scalar(3.0)));
    auto r = functional::Where(condition, a, JUST(functional::ZerosLike(x)));
    return r;
  }
};

class SoftplusGradGradFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x, const std::shared_ptr<Tensor>& dydx,
                           const double& beta, const double& threshold) const {
    auto beta_x = JUST(functional::ScalarMul(x, beta, /*inplace=*/false));
    auto condition = JUST(functional::ScalarLogicalLess(beta_x, Scalar(threshold)));
    auto sig_grad = JUST(functional::SigmoidGrad(beta_x, dydx));
    sig_grad = JUST(functional::ScalarMul(sig_grad, Scalar(beta), /*inplace=*/true));
    auto res = JUST(functional::Where(condition, sig_grad, dydx));
    return res;
  }
};

class EluGradGradFunctor {
 public:
  // ELU(x)=max(0,x)+min(0,alpha∗(exp(x)−1))
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x, const std::shared_ptr<Tensor>& dydx,
                           const double& alpha) const {
    auto condition = JUST(functional::ScalarLogicalLess(x, Scalar(0.0)));
    return functional::Where(condition, JUST(functional::EluGrad(x, dydx, alpha)),
                             JUST(functional::ZerosLike(x)));
  }
};

class CeluGradGradFunctor {
 public:
  // CELU(x)=max(0,x)+min(0,alpha∗(exp(x/alpha)−1))
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x, const std::shared_ptr<Tensor>& dydx,
                           const double& alpha) const {
    auto condition = JUST(functional::ScalarLogicalLess(x, Scalar(0)));
    auto a = JUST(functional::CeluGrad(x, dydx, alpha));
    auto b = JUST(functional::ScalarDiv(a, Scalar(alpha)));
    auto res = functional::Where(condition, b, JUST(functional::ZerosLike(x)));
    return res;
  }
};

}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<impl::SinGradGradFunctor>("SinGradGrad");
  m.add_functor<impl::CosGradGradFunctor>("CosGradGrad");
  m.add_functor<impl::SiluGradGradFunctor>("SiluGradGrad");
  m.add_functor<impl::MishGradGradFunctor>("MishGradGrad");
  m.add_functor<impl::SeluGradGradFunctor>("SeluGradGrad");
  m.add_functor<impl::SoftSignGradGradFunctor>("SoftSignGradGrad");
  m.add_functor<impl::GeluGradGradFunctor>("GeluGradGrad");
  m.add_functor<impl::HardSigmoidGradGradFunctor>("HardSigmoidGradGrad");
  m.add_functor<impl::HardSwishGradGradFunctor>("HardSwishGradGrad");
  m.add_functor<impl::SoftplusGradGradFunctor>("SoftplusGradGrad");
  m.add_functor<impl::EluGradGradFunctor>("EluGradGrad");
  m.add_functor<impl::CeluGradGradFunctor>("CeluGradGrad");
}

}  // namespace functional
}  // namespace one
}  // namespace oneflow
