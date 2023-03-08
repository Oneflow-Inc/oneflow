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

class TanGradGradFunctor {
 public:
  // dx = 1/cos^2(x), ddx = 2*sinx/cos^3(x) = tan_grad(x)*tan(x)*2
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x,
                           const std::shared_ptr<Tensor>& dydx) const {
    auto r = sequence_function(functional::Mul)
                 .then([](const std::shared_ptr<Tensor>& input) {
                   return functional::ScalarMul(Scalar(2), input);
                 })
                 .call(JUST(functional::Tan(x)), JUST(functional::TanGrad(x, dydx)));
    return r;
  }
};

class SinhGradGradFunctor {
 public:
  // dx = cosh(x), ddx = sinh(x) = cosh_grad(x)
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x,
                           const std::shared_ptr<Tensor>& dydx) const {
    return functional::CoshGrad(x, dydx);
  }
};

class CoshGradGradFunctor {
 public:
  // dx = sinh(x), ddx = cosh(x) = sinh_grad(x)
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x,
                           const std::shared_ptr<Tensor>& dydx) const {
    return functional::SinhGrad(x, dydx);
  }
};

class TanhGradGradFunctor {
 public:
  // dx = sech^2(x), ddx = -2*sech^2(x)*tanh(x) = tan_grad(x)*tanh(x)*(-2)
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x,
                           const std::shared_ptr<Tensor>& dydx) const {
    auto r = sequence_function(functional::Mul)
                 .then([](const std::shared_ptr<Tensor>& input) {
                   return functional::ScalarMul(Scalar(-2), input);
                 })
                 .call(JUST(functional::Tanh(x)), JUST(functional::TanhGrad(x, dydx)));
    return r;
  }
};

class AsinGradGradFunctor {
 public:
  // dx = 1/sqrt(1-x*x)=rsqrt(1-x*x), ddx = rsqrt_grad(1-x*x)*(-2x)
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x,
                           const std::shared_ptr<Tensor>& dydx) const {
    auto r = sequence_function(functional::Square)
                 .then([](const std::shared_ptr<Tensor>& input) {
                   return functional::ScalarSub(Scalar(1), input, /*alpha=*/1.0);
                 })
                 .then(std::bind(functional::RsqrtGrad, std::placeholders::_1, dydx))
                 .then(std::bind(functional::Mul, std::placeholders::_1, x))
                 .then([](const std::shared_ptr<Tensor>& input) {
                   return functional::ScalarMul(Scalar(-2), input);
                 })
                 .call(x);
    return r;
  }
};
class AcosGradGradFunctor {
 public:
  // dx = -1/sqrt(1-x*x)=-rsqrt(1-x*x), ddx = rsqrt_grad(1-x*x)*(2x)
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x,
                           const std::shared_ptr<Tensor>& dydx) const {
    auto r = sequence_function(functional::Square)
                 .then([](const std::shared_ptr<Tensor>& input) {
                   return functional::ScalarSub(Scalar(1), input, /*alpha=*/1.0);
                 })
                 .then(std::bind(functional::RsqrtGrad, std::placeholders::_1, dydx))
                 .then(std::bind(functional::Mul, std::placeholders::_1, x))
                 .then([](const std::shared_ptr<Tensor>& input) {
                   return functional::ScalarMul(Scalar(2), input);
                 })
                 .call(x);
    return r;
  }
};

class AtanGradGradFunctor {
 public:
  // dx = 1/(1+x*x), ddx = reci_grad(1+x*x)*(2x)
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x,
                           const std::shared_ptr<Tensor>& dydx) const {
    auto r = sequence_function(functional::Square)
                 .then([](const std::shared_ptr<Tensor>& input) {
                   return functional::ScalarAdd(Scalar(1), input, /*alpha=*/1.0);
                 })
                 .then(std::bind(functional::ReciprocalGrad, std::placeholders::_1, dydx))
                 .then(std::bind(functional::Mul, std::placeholders::_1, x))
                 .then([](const std::shared_ptr<Tensor>& input) {
                   return functional::ScalarMul(Scalar(2), input);
                 })
                 .call(x);
    return r;
  }
};

class AsinhGradGradFunctor {
 public:
  // dx = 1/sqrt(1+x*x)=rsqrt(1+x*x), ddx = rsqrt_grad(1+x*x)*(2x)
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x,
                           const std::shared_ptr<Tensor>& dydx) const {
    auto r = sequence_function(functional::Square)
                 .then([](const std::shared_ptr<Tensor>& input) {
                   return functional::ScalarAdd(Scalar(1), input, /*alpha=*/1.0);
                 })
                 .then(std::bind(functional::RsqrtGrad, std::placeholders::_1, dydx))
                 .then(std::bind(functional::Mul, std::placeholders::_1, x))
                 .then([](const std::shared_ptr<Tensor>& input) {
                   return functional::ScalarMul(Scalar(2), input);
                 })
                 .call(x);
    return r;
  }
};

class AcoshGradGradFunctor {
 public:
  // dx = 1/sqrt(x*x-1)=rsqrt(x*x-1), ddx = rsqrt_grad(x*x-1)*(2x)
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x,
                           const std::shared_ptr<Tensor>& dydx) const {
    auto r = sequence_function(functional::Square)
                 .then([](const std::shared_ptr<Tensor>& input) {
                   return functional::ScalarSub(input, Scalar(1), /*alpha=*/1.0,
                                                /*inplace=*/false);
                 })
                 .then(std::bind(functional::RsqrtGrad, std::placeholders::_1, dydx))
                 .then(std::bind(functional::Mul, std::placeholders::_1, x))
                 .then([](const std::shared_ptr<Tensor>& input) {
                   return functional::ScalarMul(Scalar(2), input);
                 })
                 .call(x);

    return r;
  }
};

class AtanhGradGradFunctor {
 public:
  // dx = 1/(1-x*x), ddx = reci_grad(1-x*x)*(-2x)
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x,
                           const std::shared_ptr<Tensor>& dydx) const {
    auto r = sequence_function(functional::Square)
                 .then([](const std::shared_ptr<Tensor>& input) {
                   return functional::ScalarSub(Scalar(1), input, /*alpha=*/1.0);
                 })
                 .then(std::bind(functional::ReciprocalGrad, std::placeholders::_1, dydx))
                 .then(std::bind(functional::Mul, std::placeholders::_1, x))
                 .then([](const std::shared_ptr<Tensor>& input) {
                   return functional::ScalarMul(Scalar(-2), input);
                 })
                 .call(x);
    return r;
  }
};

class ErfGradGradFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x,
                           const std::shared_ptr<Tensor>& dydx) const {
    return functional::ScalarMul(Scalar(-2),
                                 JUST(functional::Mul(x, JUST(functional::ErfGrad(x, dydx)))));
  }
};

class ErfcGradGradFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x,
                           const std::shared_ptr<Tensor>& dydx) const {
    return functional::ScalarMul(Scalar(-2),
                                 JUST(functional::Mul(x, JUST(functional::ErfcGrad(x, dydx)))));
  }
};

class ExpGradGradFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x,
                           const std::shared_ptr<Tensor>& dydx) const {
    return functional::ExpGrad(x, dydx);
  }
};

class Exp2GradGradFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x,
                           const std::shared_ptr<Tensor>& dydx) const {
    return functional::ScalarMul(Scalar(std::log(2)), JUST(functional::Exp2Grad(x, dydx)));
  }
};

class Expm1GradGradFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x,
                           const std::shared_ptr<Tensor>& dydx) const {
    return functional::ExpGrad(x, dydx);
  }
};

class LogGradGradFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x,
                           const std::shared_ptr<Tensor>& dydx) const {
    return functional::ReciprocalGrad(x, dydx);
  }
};

class Log2GradGradFunctor {
 public:
  // dx = 1/(x*ln2), ddx = 1/ln2 * -1/(x*x)
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x,
                           const std::shared_ptr<Tensor>& dydx) const {
    return functional::ScalarMul(Scalar(1.0 / std::log(2.0f)),
                                 JUST(functional::ReciprocalGrad(x, dydx)));
  }
};

class Log10GradGradFunctor {
 public:
  // dx = 1/(x*ln10), ddx = 1/ln10 * -1/(x*x)
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x,
                           const std::shared_ptr<Tensor>& dydx) const {
    return functional::ScalarMul(Scalar(1.0 / std::log(10.0f)),
                                 JUST(functional::ReciprocalGrad(x, dydx)));
  }
};

class Log1pGradGradFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x,
                           const std::shared_ptr<Tensor>& dydx) const {
    return functional::ReciprocalGrad(
        JUST(functional::ScalarAdd(Scalar(1), x, /*alpha=*/Scalar(1))), dydx);
  }
};

class LogSigmoidGradGradFunctor {
 public:
  // dx = exp(-x)/(1+exp(-x)), ddx = -exp(-x)/(1+exp(-x))^2 = -sigmoid_grad(x)
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x,
                           const std::shared_ptr<Tensor>& dydx) const {
    return functional::Negative(JUST(functional::SigmoidGrad(JUST(functional::Sigmoid(x)), dydx)));
  }
};

class ReciprocalGradGradFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x,
                           const std::shared_ptr<Tensor>& dydx) const {
    return functional::Negative(JUST(functional::ScalarPowGrad(x, dydx, Scalar(-2))));
  }
};

class ReciprocalNoNanGradGradFunctor {
 public:
  // dx = -pow(x,-2), ddx = -pow_grad(x,-2)
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x,
                           const std::shared_ptr<Tensor>& dydx) const {
    return functional::Negative(JUST(functional::ScalarPowGrad(x, dydx, Scalar(-2))));
  }
};

class RsqrtGradGradFunctor {
 public:
  // dx = -0.5*pow(x,-1.5), ddx = -0.5*pow_grad(x,-1.5)
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x,
                           const std::shared_ptr<Tensor>& dydx) const {
    return functional::ScalarMul(Scalar(-0.5),
                                 JUST(functional::ScalarPowGrad(x, dydx, Scalar(-1.5))));
  }
};

class SqrtGradGradFunctor {
 public:
  // dx = 0.5*pow(x,-0.5), ddx = -0.25*pow(x,-1.5) = 0.5*rsqrt_grad(x)
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x,
                           const std::shared_ptr<Tensor>& dydx) const {
    return functional::ScalarMul(Scalar(0.5), JUST(functional::RsqrtGrad(x, dydx)));
  }
};

class SquareGradGradFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x,
                           const std::shared_ptr<Tensor>& dydx) const {
    return functional::ScalarMul(2, dydx);
  }
};

class SigmoidGradGradFunctor {
 public:
  // dy = y * (1 - y), ddy = 1 - 2*y
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& y,
                           const std::shared_ptr<Tensor>& dydx) const {
    return functional::Mul(JUST(functional::ScalarSub(1, y, /*alpha=*/2)), dydx);
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
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& y, const std::shared_ptr<Tensor>& dydx,
                           const double& alpha) const {
    auto condition = JUST(functional::ScalarLogicalLess(y, Scalar(0)));
    auto r = functional::Where(condition, JUST(functional::ScalarDiv(dydx, alpha)),
                               JUST(functional::ZerosLike(y)));
    return r;
  }
};

class MaxPoolNdGradGradFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& dydx,
                           const std::shared_ptr<Tensor>& indices, const int ndims) const {
    if (indices->nelement()) {
      Shape view_shape(indices->shape()->begin(), indices->shape()->end() - ndims);
      view_shape.push_back(-1);
      auto indices_view = JUST(functional::Reshape(indices, view_shape));
      auto outgrad_view = JUST(functional::Reshape(dydx, view_shape));
      return functional::sequence_function(functional::DimGather)
          .then(std::bind(functional::Reshape, std::placeholders::_1, *indices->shape()))
          .call(outgrad_view, -1, indices_view, /*sparse_grad=*/false);
    } else {
      // empty inputs, return 0size tensor
      return functional::ZerosLike(indices);
    }
  }
};

class MishGradGradFunctor {
 public:
  // y = x ∗ tanh(softplus(x))
  // ddx = grad_tsp * sig * (2 + x * (1 + (-1 - 2 * tsp) * sig)), sig equal grad_sp here
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x,
                           const std::shared_ptr<Tensor>& dydx) const {
    const auto sig = JUST(functional::Sigmoid(x));
    const auto sp = JUST(functional::Log1p(JUST(functional::Exp(x))));
    const auto grad_tsp = JUST(functional::TanhGrad(sp, dydx));

    auto r = functional::sequence_function(functional::Tanh)
                 .then([](const std::shared_ptr<Tensor>& input) {
                   return functional::ScalarAdd(-1, input, /*alpha=*/-2);
                 })
                 .then(std::bind(functional::Mul, std::placeholders::_1, sig))
                 .then([](const std::shared_ptr<Tensor>& input) {
                   return functional::ScalarAdd(1, input, /*alpha=*/1);
                 })
                 .then(std::bind(functional::Mul, std::placeholders::_1, x))
                 .then([](const std::shared_ptr<Tensor>& input) {
                   return functional::ScalarAdd(2, input, /*alpha=*/1);
                 })
                 .then(std::bind(functional::Mul, std::placeholders::_1, sig))
                 .then(std::bind(functional::Mul, std::placeholders::_1, grad_tsp))
                 .call(sp);
    return r;
  }
};

class GeluGradGradFunctor {
 public:
  // y = gussian(x) = 0.5 * x * (1.0 + erf(sqrt(0.5) * x));
  // dx = 0.5 * (1.0 + erf(sqrt(0.5)*x) + x * coef * exp(-0.5*x*x)) * dy), coef = sqrt(-2.0/pi)
  // ddx = coef * grad1 * grad2 * flow.exp(t) * (1+t), t = -0.5*x*x
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x,
                           const std::shared_ptr<Tensor>& dydx) const {
    const auto& tmp = JUST(functional::ScalarMul(-0.5, JUST(functional::Square(x))));
    const auto& tmp_add_one = JUST(functional::ScalarAdd(1, tmp, 1));
    const Scalar coef = std::sqrt(2.0 / std::acos(-1.0));

    auto r = functional::sequence_function(functional::Exp)
                 .then(std::bind(functional::Mul, std::placeholders::_1, tmp_add_one))
                 .then(std::bind(functional::Mul, std::placeholders::_1, dydx))
                 .then([&coef](const std::shared_ptr<Tensor>& input) {
                   return functional::ScalarMul(coef, input);
                 })
                 .call(tmp);
    return r;
  }
};
}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<impl::SinGradGradFunctor>("SinGradGrad");
  m.add_functor<impl::CosGradGradFunctor>("CosGradGrad");
  m.add_functor<impl::TanGradGradFunctor>("TanGradGrad");
  m.add_functor<impl::SinhGradGradFunctor>("SinhGradGrad");
  m.add_functor<impl::CoshGradGradFunctor>("CoshGradGrad");
  m.add_functor<impl::TanhGradGradFunctor>("TanhGradGrad");
  m.add_functor<impl::AsinGradGradFunctor>("AsinGradGrad");
  m.add_functor<impl::AcosGradGradFunctor>("AcosGradGrad");
  m.add_functor<impl::AtanGradGradFunctor>("AtanGradGrad");
  m.add_functor<impl::AsinhGradGradFunctor>("AsinhGradGrad");
  m.add_functor<impl::AcoshGradGradFunctor>("AcoshGradGrad");
  m.add_functor<impl::AtanhGradGradFunctor>("AtanhGradGrad");
  m.add_functor<impl::ErfGradGradFunctor>("ErfGradGrad");
  m.add_functor<impl::ErfcGradGradFunctor>("ErfcGradGrad");
  m.add_functor<impl::ExpGradGradFunctor>("ExpGradGrad");
  m.add_functor<impl::Exp2GradGradFunctor>("Exp2GradGrad");
  m.add_functor<impl::Expm1GradGradFunctor>("Expm1GradGrad");
  m.add_functor<impl::LogGradGradFunctor>("LogGradGrad");
  m.add_functor<impl::Log2GradGradFunctor>("Log2GradGrad");
  m.add_functor<impl::Log10GradGradFunctor>("Log10GradGrad");
  m.add_functor<impl::Log1pGradGradFunctor>("Log1pGradGrad");
  m.add_functor<impl::LogSigmoidGradGradFunctor>("LogSigmoidGradGrad");
  m.add_functor<impl::ReciprocalGradGradFunctor>("ReciprocalGradGrad");
  m.add_functor<impl::ReciprocalNoNanGradGradFunctor>("ReciprocalNoNanGradGrad");
  m.add_functor<impl::RsqrtGradGradFunctor>("RsqrtGradGrad");
  m.add_functor<impl::SqrtGradGradFunctor>("SqrtGradGrad");
  m.add_functor<impl::SquareGradGradFunctor>("SquareGradGrad");
  m.add_functor<impl::SigmoidGradGradFunctor>("SigmoidGradGrad");
  m.add_functor<impl::SiluGradGradFunctor>("SiluGradGrad");
  m.add_functor<impl::SeluGradGradFunctor>("SeluGradGrad");
  m.add_functor<impl::SoftSignGradGradFunctor>("SoftSignGradGrad");
  m.add_functor<impl::HardSigmoidGradGradFunctor>("HardSigmoidGradGrad");
  m.add_functor<impl::HardSwishGradGradFunctor>("HardSwishGradGrad");
  m.add_functor<impl::SoftplusGradGradFunctor>("SoftplusGradGrad");
  m.add_functor<impl::EluGradGradFunctor>("EluGradGrad");
  m.add_functor<impl::CeluGradGradFunctor>("CeluGradGrad");
  m.add_functor<impl::MaxPoolNdGradGradFunctor>("MaxPoolNdGradGrad");
  m.add_functor<impl::MishGradGradFunctor>("MishGradGrad");
  m.add_functor<impl::GeluGradGradFunctor>("GeluGradGrad");
}

}  // namespace functional
}  // namespace one
}  // namespace oneflow
