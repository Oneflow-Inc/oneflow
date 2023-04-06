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
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/common/error.h"
#include "oneflow/core/common/scalar.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/functional/functional_api.yaml.h"
#include "oneflow/core/functional/function_library.h"
#include "oneflow/core/functional/impl/unary_functor.h"
#include "oneflow/core/functional/impl/binary_functor.h"
#include "oneflow/core/functional/sequence_function.h"
#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/mutable_attr_map.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_util.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/autograd/autograd_mode.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/user/kernels/distributions/common.h"
#include "oneflow/user/kernels/random_seed_util.h"

namespace oneflow {
namespace one {
namespace functional {

namespace impl {

class ReluFunctor {
 public:
  ReluFunctor() { op_ = CHECK_JUST(one::OpBuilder("relu").Input("x", 1).Output("y", 1).Build()); }
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x, bool inplace) const {
    if (inplace) {
      JUST(CheckInplaceValid(x));
      std::shared_ptr<TensorTuple> outputs = std::make_shared<TensorTuple>(1);
      outputs->at(0) = x;
      JUST(OpInterpUtil::Dispatch(*op_, {x}, outputs.get(), AttrMap{}));
      return outputs->at(0);
    } else {
      return OpInterpUtil::Dispatch<Tensor>(*op_, {x});
    }
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ReluGradFunctor : public BinaryFunctor {
 public:
  ReluGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("relu_grad").Input("dy").Input("y").Output("dx").Build());
  }
};

class PReluFunctor {
 public:
  PReluFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("prelu").Input("x").Input("alpha").Output("y").Build());
  }

  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x,
                           const std::shared_ptr<Tensor>& alpha) const {
    int num_params = alpha->dim(0);
    CHECK_OR_RETURN(((num_params == 1) || (num_params == x->shape()->At(1))))
        << Error::RuntimeError() << "num_parameters in prelu must be 1 or " << x->shape()->At(1);
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x, alpha});
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class PReluGradFunctor {
 public:
  PReluGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("prelu_grad")
                         .Input("dy")
                         .Input("x")
                         .Input("alpha")
                         .Output("dx")
                         .Output("alpha_diff")
                         .Build());
  }
  Maybe<TensorTuple> operator()(const std::shared_ptr<Tensor>& dy, const std::shared_ptr<Tensor>& x,
                                const std::shared_ptr<Tensor>& alpha) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("alpha_requires_grad");
    attrs.SetAllAttrs(alpha->requires_grad());
    return OpInterpUtil::Dispatch<one::TensorTuple>(*op_, {dy, x, alpha}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class HardTanhFunctor {
 public:
  HardTanhFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("hardtanh").Input("in").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const double& min_val,
                           const double& max_val) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("min_val", "max_val");
    attrs.SetAllAttrs(min_val, max_val);
    return OpInterpUtil::Dispatch<one::Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class HardTanhGradFunctor {
 public:
  HardTanhGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("hardtanh_grad").Input("y").Input("dy").Output("dx").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& y,
                           const std::shared_ptr<one::Tensor>& dy, const double& min_val,
                           const double& max_val) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("min_val", "max_val");
    attrs.SetAllAttrs(min_val, max_val);
    return OpInterpUtil::Dispatch<one::Tensor>(*op_, {y, dy}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class EluFunctor {
 public:
  EluFunctor() { op_ = CHECK_JUST(one::OpBuilder("elu").Input("in").Output("out").Build()); }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const double& alpha) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("alpha");
    attrs.SetAllAttrs(alpha);
    return OpInterpUtil::Dispatch<one::Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class EluGradFunctor {
 public:
  EluGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("elu_grad").Input("x").Input("dy").Output("dx").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& dy, const double& alpha) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("alpha");
    attrs.SetAllAttrs(alpha);
    return OpInterpUtil::Dispatch<one::Tensor>(*op_, {x, dy}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class CeluFunctor {
 public:
  CeluFunctor() { op_ = CHECK_JUST(one::OpBuilder("celu").Input("in").Output("out").Build()); }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const double& alpha,
                           bool inplace) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("alpha");
    attrs.SetAllAttrs(alpha);
    if (inplace) {
      JUST(CheckInplaceValid(x));
      std::shared_ptr<TensorTuple> outputs = std::make_shared<TensorTuple>(1);
      (*outputs)[0] = x;
      JUST(OpInterpUtil::Dispatch(*op_, {x}, outputs.get(), attrs));
      return outputs->at(0);
    } else {
      return OpInterpUtil::Dispatch<one::Tensor>(*op_, {x}, attrs);
    }
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class CeluGradFunctor {
 public:
  CeluGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("celu_grad").Input("y").Input("dy").Output("dx").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& y,
                           const std::shared_ptr<one::Tensor>& dy, const double& alpha) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("alpha");
    attrs.SetAllAttrs(alpha);
    return OpInterpUtil::Dispatch<one::Tensor>(*op_, {y, dy}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class GeluFunctor : public UnaryFunctor {
 public:
  GeluFunctor() { op_ = CHECK_JUST(one::OpBuilder("gelu").Input("in").Output("out").Build()); }
};

class GeluGradFunctor : public BinaryFunctor {
 public:
  GeluGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("gelu_grad").Input("dy").Input("x").Output("dx").Build());
  }
};

class FastGeluFunctor : public UnaryFunctor {
 public:
  FastGeluFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("fast_gelu").Input("in").Output("out").Build());
  }
};

class FastGeluGradFunctor : public BinaryFunctor {
 public:
  FastGeluGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("fast_gelu_grad").Input("dy").Input("x").Output("dx").Build());
  }
};

class QuickGeluFunctor : public UnaryFunctor {
 public:
  QuickGeluFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("quick_gelu").Input("x").Output("y").Build());
  }
};

class QuickGeluGradFunctor : public BinaryFunctor {
 public:
  QuickGeluGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("quick_gelu_grad").Input("dy").Input("x").Output("dx").Build());
  }
};

class GluFunctor {
 public:
  GluFunctor() {}
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input, int64_t dim) const {
    const auto ndim = input->ndim();
    CHECK_GT_OR_RETURN(ndim, 0) << Error::RuntimeError()
                                << "glu does not support scalars because halving size must be even";
    dim = JUST(maybe_wrap_dim(dim, ndim));
    if (dim < 0) { dim += ndim; }
    int64_t nc = input->dim(dim);
    CHECK_EQ_OR_RETURN(nc % 2, 0) << Error::RuntimeError()
                                  << "Halving dimension must be even, but dimension " << dim
                                  << " is size " << nc;
    nc = nc / 2;
    std::vector<int64_t> split_sizes(2, nc);
    const auto split_x = JUST(SplitWithSize(input, split_sizes, dim));
    return sequence_function(functional::Sigmoid)
        .then(std::bind(functional::Mul, (*split_x)[0], std::placeholders::_1))
        .call((*split_x)[1]);
  }
};

class HardSigmoidFunctor {
 public:
  HardSigmoidFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("hardsigmoid").Input("in").Output("out").Build());
  }

  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& input, bool inplace) const {
    if (inplace) {
      JUST(CheckInplaceValid(input));
      std::shared_ptr<TensorTuple> outputs = std::make_shared<TensorTuple>(1);
      outputs->at(0) = input;
      JUST(OpInterpUtil::Dispatch(*op_, {input}, outputs.get(), AttrMap{}));
      return outputs->at(0);
    } else {
      return OpInterpUtil::Dispatch<Tensor>(*op_, {input});
    }
  }

 private:
  std::shared_ptr<OpExpr> op_;
};
class HardSigmoidGradFunctor : public BinaryFunctor {
 public:
  HardSigmoidGradFunctor() {
    op_ =
        CHECK_JUST(one::OpBuilder("hardsigmoid_grad").Input("dy").Input("x").Output("dx").Build());
  }
};

class HardShrinkFunctor {
 public:
  HardShrinkFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("hardshrink").Input("in").Output("out").Build());
  }

  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x, const double& lambd,
                           bool inplace) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("lambd");
    attrs.SetAllAttrs(lambd);
    if (inplace) {
      JUST(CheckInplaceValid(x));
      std::shared_ptr<TensorTuple> outputs = std::make_shared<TensorTuple>(1);
      JUST(oneflow::VectorAt(*outputs, 0)) = x;
      JUST(OpInterpUtil::Dispatch(*op_, {x}, outputs.get(), attrs));
      return JUST(oneflow::VectorAt(*outputs, 0));
    } else {
      return OpInterpUtil::Dispatch<one::Tensor>(*op_, {x}, attrs);
    }
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class HardShrinkGradFunctor {
 public:
  HardShrinkGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("hardshrink_grad").Input("y").Input("dy").Output("dx").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& y, const std::shared_ptr<Tensor>& dy,
                           const double& lambd) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("lambd");
    attrs.SetAllAttrs(lambd);
    return OpInterpUtil::Dispatch<one::Tensor>(*op_, {y, dy}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class SoftmaxFunctorBase {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const Optional<int64_t>& dim) const {
    const auto input_shape = input->shape();
    const int64_t num_axes = input_shape->NumAxes();

    const auto get_dim = [num_axes]() -> int64_t {
      const int64_t ndim = num_axes;
      if (ndim == 0 || ndim == 1 || ndim == 3) {
        return 0;
      } else {
        return 1;
      }
    };

    int64_t dim_ = dim ? JUST(dim) : get_dim();
    dim_ = JUST(maybe_wrap_dim(dim_, num_axes));
    if (dim_ != num_axes - 1) {
      std::vector<int> input_perm(input_shape->dim_vec().size(), 0);
      for (size_t i = 1; i < input_perm.size(); ++i) { input_perm[i] = i; }
      input_perm[dim_] = input_perm[input_perm.size() - 1];
      input_perm[input_perm.size() - 1] = dim_;

      return sequence_function(functional::Transpose)
          .then([&](const std::shared_ptr<one::Tensor>& x) {
            return OpInterpUtil::Dispatch<Tensor>(*op_, {x});
          })
          .then(std::bind(functional::Transpose, std::placeholders::_1, input_perm))
          .call(input, input_perm);
    }

    return OpInterpUtil::Dispatch<Tensor>(*op_, {input});
  }

 protected:
  SoftmaxFunctorBase() = default;
  virtual ~SoftmaxFunctorBase() = default;

  std::shared_ptr<OpExpr> op_;
};

class SoftmaxFunctor : public SoftmaxFunctorBase {
 public:
  SoftmaxFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("softmax").Input("in").Output("out").Build());
  }
};

class SoftmaxGradFunctor {
 public:
  SoftmaxGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("softmax_grad").Input("y").Input("dy").Output("dx").Build());
  }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& y) const {
    return OpInterpUtil::Dispatch<Tensor>(*op_, {y, dy});
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class LogSoftmaxFunctor : public SoftmaxFunctorBase {
 public:
  LogSoftmaxFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("log_softmax").Input("in").Output("prob").Build());
  }
};

class LogSoftmaxGradFunctor {
 public:
  LogSoftmaxGradFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("log_softmax_grad").Input("prob").Input("dy").Output("dx").Build());
  }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& y) const {
    return OpInterpUtil::Dispatch<Tensor>(*op_, {y, dy});
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class GumbelSoftmaxFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& in, const double& tau,
                           const Optional<int64_t>& dim, bool hard,
                           const Optional<one::Generator>& generator) const {
    auto in_shape = in->shape();
    auto device = JUST(in->device());
    auto dtype = in->dtype();
    const int64_t num_axes = in_shape->NumAxes();

    const auto gen = generator.value_or(JUST(one::DefaultAutoGenerator()));
    auto random_tensor =
        JUST(functional::Rand(*in_shape.get(), dtype, device, gen, /*requires_grad=*/false));
    auto gumbel_noise_tensor = JUST(functional::ScalarSub(
        Scalar(0.0),
        JUST(functional::Log(JUST(functional::ScalarSub(
            Scalar(0.0), JUST(functional::Log(random_tensor)), /*alpha=*/1.0)))),
        /*alpha=*/1.0));
    auto gumbel_in_tensor = JUST(functional::ScalarDiv(
        JUST(functional::Add(in, gumbel_noise_tensor, /*alpha=*/1.0, /*inplace=*/false)),
        Scalar(tau)));

    auto out_soft = JUST(functional::Softmax(gumbel_in_tensor, dim));
    if (hard) {
      const auto get_dim = [num_axes]() -> int64_t {
        const int64_t ndim = num_axes;
        if (ndim == 0 || ndim == 1 || ndim == 3) {
          return 0;
        } else {
          return 1;
        }
      };

      int64_t dim_ = dim ? JUST(dim) : get_dim();
      dim_ = JUST(maybe_wrap_dim(dim_, num_axes));
      auto out_max = JUST(functional::ArgMax(out_soft, dim_, /*keepdim=*/true, dtype));
      auto index =
          JUST(functional::To(out_max, JUST(DType::Get(DataType::kInt64)), /*copy=*/false));
      auto zero = JUST(functional::ZerosLike(out_soft));
      auto out_hard =
          JUST(functional::DimScatterUpdateScalar(zero, dim_, index, 1.0, /*inplace=*/false));

      auto out_hard_has_grad =
          functional::Add(JUST(functional::Sub(out_hard, JUST(out_soft->detach()), /*alpha=*/1.0,
                                               /*inplace=*/false)),
                          out_soft, /*alpha=*/1.0, /*inplace=*/false);
      return out_hard_has_grad;
    } else {
      return out_soft;
    }
  }
};

class HardSwishFunctor : public UnaryFunctor {
 public:
  HardSwishFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("hardswish").Input("in").Output("out").Build());
  }
};

class HardSwishGradFunctor : public BinaryFunctor {
 public:
  HardSwishGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("hardswish_grad").Input("dy").Input("x").Output("dx").Build());
  }
};

class LeakyReluFunctor {
 public:
  LeakyReluFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("leaky_relu").Input("x").Output("y").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const float& alpha,
                           bool inplace) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("alpha");
    attrs.SetAllAttrs(alpha);
    if (inplace) {
      JUST(CheckInplaceValid(x));
      std::shared_ptr<TensorTuple> outputs = std::make_shared<TensorTuple>(1);
      JUST(oneflow::VectorAt(*outputs, 0)) = x;
      JUST(OpInterpUtil::Dispatch(*op_, {x}, outputs.get(), attrs));
      return JUST(oneflow::VectorAt(*outputs, 0));
    } else {
      return OpInterpUtil::Dispatch<one::Tensor>(*op_, {x}, attrs);
    }
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class LeakyReluGradFunctor {
 public:
  LeakyReluGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("leaky_relu_grad").Input("x").Input("dy").Output("dx").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& dy, const float& alpha) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("alpha");
    attrs.SetAllAttrs(alpha);
    return OpInterpUtil::Dispatch<one::Tensor>(*op_, {x, dy}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class RReluFunctor {
 public:
  RReluFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("rrelu").Input("in").Output("output").Output("noise_data").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const float& lower,
                           const float& upper, bool training, bool inplace) const {
    if (!training) { return JUST(functional::LeakyRelu(x, ((lower + upper) / 2), inplace)); }

    auto gen = JUST(
        GetGeneratorForLazyOrGlobal(JUST(one::DefaultAutoGenerator()), LazyMode::is_enabled(), x));
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("seed", "lower", "upper", "training");
    attrs.SetAllAttrs(static_cast<int64_t>(gen->current_seed()), lower, upper, training);
    const auto& state = std::make_shared<DistributionKernelState>(gen);

    OpExprInterpContext ctx(attrs, state);
    std::shared_ptr<TensorTuple> outputs = std::make_shared<TensorTuple>(2);
    if (inplace) {
      JUST(CheckInplaceValid(x));
      outputs->at(0) = x;
    }
    JUST(OpInterpUtil::Dispatch(*op_, {x}, outputs.get(), ctx));
    return outputs->at(0);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class RReluInplaceFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const float& lower,
                           const float& upper, bool training) const {
    return JUST(functional::RRelu(x, lower, upper, training, true /*inplace*/));
  }
};

class SoftplusFunctor {
 public:
  SoftplusFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("softplus").Input("in").Output("out").Build());
  }

  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x, const double& beta,
                           const double& threshold) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("beta", "threshold");
    attrs.SetAllAttrs(beta, threshold);
    return OpInterpUtil::Dispatch<one::Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class SoftplusGradFunctor {
 public:
  SoftplusGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("softplus_grad").Input("x").Input("dy").Output("dx").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x, const std::shared_ptr<Tensor>& dy,
                           const double& beta, const double& threshold) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("beta", "threshold");
    attrs.SetAllAttrs(beta, threshold);
    return OpInterpUtil::Dispatch<one::Tensor>(*op_, {x, dy}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class SiluFunctor : public UnaryFunctor {
 public:
  SiluFunctor() { op_ = CHECK_JUST(one::OpBuilder("silu").Input("in").Output("out").Build()); }
};

class SiluGradFunctor : public BinaryFunctor {
 public:
  SiluGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("silu_grad").Input("dy").Input("x").Output("dx").Build());
  }
};

class MishFunctor : public UnaryFunctor {
 public:
  MishFunctor() { op_ = CHECK_JUST(one::OpBuilder("mish").Input("in").Output("out").Build()); }
};

class MishGradFunctor : public BinaryFunctor {
 public:
  MishGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("mish_grad").Input("dy").Input("x").Output("dx").Build());
  }
};

class SeluFunctor : public UnaryFunctor {
 public:
  SeluFunctor() { op_ = CHECK_JUST(one::OpBuilder("selu").Input("in").Output("out").Build()); }
};

class SeluGradFunctor : public BinaryFunctor {
 public:
  SeluGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("selu_grad").Input("dy").Input("x").Output("dx").Build());
  }
};

class SoftSignFunctor : public UnaryFunctor {
 public:
  SoftSignFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("softsign").Input("in").Output("out").Build());
  }
};

class SoftSignGradFunctor : public BinaryFunctor {
 public:
  SoftSignGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("softsign_grad").Input("dy").Input("x").Output("dx").Build());
  }
};

class SoftShrinkFunctor {
 public:
  SoftShrinkFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("softshrink").Input("in").Output("out").Build());
  }

  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x, const double& alpha,
                           bool inplace) const {
    CHECK_GE_OR_RETURN(alpha, 0) << Error::RuntimeError()
                                 << "alpha must be greater or equal to 0, but found to be " << alpha
                                 << ".";
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("alpha");
    attrs.SetAllAttrs(alpha);
    if (inplace) {
      JUST(CheckInplaceValid(x));
      std::shared_ptr<TensorTuple> outputs = std::make_shared<TensorTuple>(1);
      JUST(oneflow::VectorAt(*outputs, 0)) = x;
      JUST(OpInterpUtil::Dispatch(*op_, {x}, outputs.get(), attrs));
      return JUST(oneflow::VectorAt(*outputs, 0));
    } else {
      return OpInterpUtil::Dispatch<one::Tensor>(*op_, {x}, attrs);
    }
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ThresholdFunctor {
 public:
  ThresholdFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("threshold").Input("in").Output("out").Build());
  }

  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x, const double& threshold,
                           const double& value) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("threshold_val", "value");
    attrs.SetAllAttrs(threshold, value);
    return OpInterpUtil::Dispatch<one::Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ThresholdGradFunctor {
 public:
  ThresholdGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("threshold_grad").Input("x").Input("dy").Output("dx").Build());
  }

  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x, const std::shared_ptr<Tensor>& dy,
                           const double& threshold) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("threshold_val");
    attrs.SetAllAttrs(threshold);
    return OpInterpUtil::Dispatch<one::Tensor>(*op_, {x, dy}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class SoftShrinkGradFunctor {
 public:
  SoftShrinkGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("softshrink_grad").Input("y").Input("dy").Output("dx").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& y, const std::shared_ptr<Tensor>& dy,
                           const double& alpha) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("alpha");
    attrs.SetAllAttrs(alpha);
    return OpInterpUtil::Dispatch<one::Tensor>(*op_, {y, dy}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class FracFunctor {
 public:
  FracFunctor() { op_ = CHECK_JUST(one::OpBuilder("frac").Input("x").Output("y").Build()); }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x) const {
    return OpInterpUtil::Dispatch<one::Tensor>(*op_, {x});
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class FracInplaceFunctor {
 public:
  FracInplaceFunctor() { op_ = CHECK_JUST(one::OpBuilder("frac").Input("x").Output("y").Build()); }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x) const {
    JUST(CheckInplaceValid(x));
    std::shared_ptr<TensorTuple> outputs = std::make_shared<TensorTuple>(1);
    outputs->at(0) = x;
    JUST(OpInterpUtil::Dispatch(*op_, {x}, outputs.get(), AttrMap{}));
    return outputs->at(0);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<impl::FracFunctor>("Frac");
  m.add_functor<impl::FracInplaceFunctor>("FracInplace");
  m.add_functor<impl::ReluFunctor>("Relu");
  m.add_functor<impl::ReluGradFunctor>("ReluGrad");
  m.add_functor<impl::PReluFunctor>("PRelu");
  m.add_functor<impl::PReluGradFunctor>("PReluGrad");
  m.add_functor<impl::HardTanhFunctor>("HardTanh");
  m.add_functor<impl::HardTanhGradFunctor>("HardTanhGrad");
  m.add_functor<impl::EluFunctor>("Elu");
  m.add_functor<impl::EluGradFunctor>("EluGrad");
  m.add_functor<impl::CeluFunctor>("Celu");
  m.add_functor<impl::CeluGradFunctor>("CeluGrad");
  m.add_functor<impl::GeluFunctor>("Gelu");
  m.add_functor<impl::GeluGradFunctor>("GeluGrad");
  m.add_functor<impl::FastGeluFunctor>("FastGelu");
  m.add_functor<impl::FastGeluGradFunctor>("FastGeluGrad");
  m.add_functor<impl::QuickGeluFunctor>("QuickGelu");
  m.add_functor<impl::QuickGeluGradFunctor>("QuickGeluGrad");
  m.add_functor<impl::GluFunctor>("Glu");
  m.add_functor<impl::HardSigmoidFunctor>("HardSigmoid");
  m.add_functor<impl::HardSigmoidGradFunctor>("HardSigmoidGrad");
  m.add_functor<impl::HardShrinkFunctor>("HardShrink");
  m.add_functor<impl::HardShrinkGradFunctor>("HardShrinkGrad");
  m.add_functor<impl::SoftmaxFunctor>("Softmax");
  m.add_functor<impl::SoftmaxGradFunctor>("SoftmaxGrad");
  m.add_functor<impl::LogSoftmaxFunctor>("LogSoftmax");
  m.add_functor<impl::LogSoftmaxGradFunctor>("LogSoftmaxGrad");
  m.add_functor<impl::GumbelSoftmaxFunctor>("GumbelSoftmax");
  m.add_functor<impl::HardSwishFunctor>("HardSwish");
  m.add_functor<impl::HardSwishGradFunctor>("HardSwishGrad");
  m.add_functor<impl::LeakyReluFunctor>("LeakyRelu");
  m.add_functor<impl::LeakyReluGradFunctor>("LeakyReluGrad");
  m.add_functor<impl::RReluFunctor>("RRelu");
  m.add_functor<impl::RReluInplaceFunctor>("RReluInplace");
  m.add_functor<impl::SoftplusFunctor>("Softplus");
  m.add_functor<impl::SoftplusGradFunctor>("SoftplusGrad");
  m.add_functor<impl::SiluFunctor>("Silu");
  m.add_functor<impl::SiluGradFunctor>("SiluGrad");
  m.add_functor<impl::MishFunctor>("Mish");
  m.add_functor<impl::MishGradFunctor>("MishGrad");
  m.add_functor<impl::SeluFunctor>("Selu");
  m.add_functor<impl::SeluGradFunctor>("SeluGrad");
  m.add_functor<impl::SoftSignFunctor>("SoftSign");
  m.add_functor<impl::SoftSignGradFunctor>("SoftSignGrad");
  m.add_functor<impl::ThresholdFunctor>("Threshold");
  m.add_functor<impl::ThresholdGradFunctor>("ThresholdGrad");
  m.add_functor<impl::SoftShrinkFunctor>("SoftShrink");
  m.add_functor<impl::SoftShrinkGradFunctor>("SoftShrinkGrad");
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow
