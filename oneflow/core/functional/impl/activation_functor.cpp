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

#include "oneflow/core/functional/impl/unary_functor.h"
#include "oneflow/core/functional/impl/binary_functor.h"

#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/functional/function_library.h"
#include "oneflow/core/functional/scalar.h"
#include "oneflow/core/autograd/autograd_mode.h"

namespace oneflow {
namespace one {
namespace functional {

namespace impl {

class ReluFunctor {
 public:
  ReluFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("relu").Input("in", 1).Output("out", 1).Build());
  }
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

class PReluFunctor : public BinaryFunctor {
 public:
  PReluFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("prelu").Input("x").Input("alpha").Output("y").Build());
  }
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

class HardSigmoidFunctor : public UnaryFunctor {
 public:
  HardSigmoidFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("hardsigmoid").Input("in").Output("out").Build());
  }
};

class HardSigmoidGradFunctor : public BinaryFunctor {
 public:
  HardSigmoidGradFunctor() {
    op_ =
        CHECK_JUST(one::OpBuilder("hardsigmoid_grad").Input("dy").Input("x").Output("dx").Build());
  }
};

class SoftmaxFunctor : public UnaryFunctor {
 public:
  SoftmaxFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("softmax").Input("in").Output("out").Build());
  }
};

class LogSoftmaxFunctor : public UnaryFunctor {
 public:
  LogSoftmaxFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("logsoftmax").Input("in").Output("out").Output("prob").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& logits) const {
    return OpInterpUtil::Dispatch<Tensor>(*op_, {logits});
  }

 private:
  std::shared_ptr<OpExpr> op_;
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
  m.add_functor<impl::GeluFunctor>("Gelu");
  m.add_functor<impl::GeluGradFunctor>("GeluGrad");
  m.add_functor<impl::HardSigmoidFunctor>("HardSigmoid");
  m.add_functor<impl::HardSigmoidGradFunctor>("HardSigmoidGrad");
  m.add_functor<impl::SoftmaxFunctor>("Softmax");
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
