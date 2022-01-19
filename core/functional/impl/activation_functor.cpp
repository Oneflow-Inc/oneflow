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
#include "oneflow/core/common/scalar.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/functional/impl/unary_functor.h"
#include "oneflow/core/functional/impl/binary_functor.h"
#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/functional/function_library.h"
#include "oneflow/core/autograd/autograd_mode.h"
#include "oneflow/core/functional/sequence_function.h"

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
        << "num_parameters in prelu must be 1 or " << x->shape()->At(1);
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
    return OpInterpUtil::Dispatch<one::TensorTuple>(*op_, {dy, x, alpha});
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<double>("min_val", min_val));
    JUST(attrs.SetAttr<double>("max_val", max_val));
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<double>("min_val", min_val));
    JUST(attrs.SetAttr<double>("max_val", max_val));
    return OpInterpUtil::Dispatch<one::Tensor>(*op_, {y, dy}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class EluFunctor {
 public:
  EluFunctor() { op_ = CHECK_JUST(one::OpBuilder("elu").Input("in").Output("out").Build()); }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const double& alpha) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<double>("alpha", alpha));
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<double>("alpha", alpha));
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<double>("alpha", alpha));
    if (inplace) {
      JUST(CheckInplaceValid(x));
      std::shared_ptr<TensorTuple> outputs = std::make_shared<TensorTuple>(1);
      outputs->at(0) = x;
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
    op_ = CHECK_JUST(one::OpBuilder("celu_grad").Input("x").Input("dy").Output("dx").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& dy, const double& alpha) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<double>("alpha", alpha));
    return OpInterpUtil::Dispatch<one::Tensor>(*op_, {x, dy}, attrs);
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

class GluFunctor {
 public:
  GluFunctor() {}
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input, int64_t dim) const {
    auto ndim = input->ndim();
    CHECK_GT_OR_RETURN(ndim, 0) << "glu does not support 0-dimensional tensors";
    CHECK_OR_RETURN(dim >= -ndim && dim < ndim)
        << ", Dimension out of range (expected to be in range of [" << -ndim << ", " << ndim - 1
        << "], but got " << dim << ")";
    if (dim < 0) { dim += ndim; }
    int64_t nc = input->dim(dim);
    CHECK_EQ_OR_RETURN(nc % 2, 0) << "Halving dimension must be even, but dimension " << dim
                                  << " is size " << nc;
    nc = nc / 2;
    std::vector<int64_t> split_sizes(2, nc);
    const auto split_x = JUST(SplitWithSize(input, split_sizes, dim));
    return sequence_function(functional::Sigmoid)
        .then(std::bind(functional::Mul, split_x->at(0), std::placeholders::_1))
        .call(split_x->at(1));
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
    if (dim_ < 0) { dim_ += num_axes; }

    CHECK_GE_OR_RETURN(dim_, 0);
    CHECK_LT_OR_RETURN(dim_, num_axes);

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
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const float& alpha) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<float>("alpha", alpha));
    return OpInterpUtil::Dispatch<one::Tensor>(*op_, {x}, attrs);
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<float>("alpha", alpha));
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

}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) {
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
  m.add_functor<impl::GluFunctor>("Glu");
  m.add_functor<impl::HardSigmoidFunctor>("HardSigmoid");
  m.add_functor<impl::HardSigmoidGradFunctor>("HardSigmoidGrad");
  m.add_functor<impl::SoftmaxFunctor>("Softmax");
  m.add_functor<impl::SoftmaxGradFunctor>("SoftmaxGrad");
  m.add_functor<impl::LogSoftmaxFunctor>("LogSoftmax");
  m.add_functor<impl::HardSwishFunctor>("HardSwish");
  m.add_functor<impl::HardSwishGradFunctor>("HardSwishGrad");
  m.add_functor<impl::LeakyReluFunctor>("LeakyRelu");
  m.add_functor<impl::LeakyReluGradFunctor>("LeakyReluGrad");
  m.add_functor<impl::SiluFunctor>("Silu");
  m.add_functor<impl::SiluGradFunctor>("SiluGrad");
  m.add_functor<impl::MishFunctor>("Mish");
  m.add_functor<impl::MishGradFunctor>("MishGrad");
  m.add_functor<impl::SeluFunctor>("Selu");
  m.add_functor<impl::SeluGradFunctor>("SeluGrad");
  m.add_functor<impl::SoftSignFunctor>("SoftSign");
  m.add_functor<impl::SoftSignGradFunctor>("SoftSignGrad");
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow
