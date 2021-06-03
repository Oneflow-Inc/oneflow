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

namespace oneflow {
namespace one {
namespace functional {

namespace impl {

class ReluFunctor : public UnaryFunctor {
 public:
  ReluFunctor() { op_ = CHECK_JUST(one::OpBuilder("relu").Input("in").Output("out").Build()); }
};

class PReluFunctor : public BinaryFunctor {
 public:
  PReluFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("prelu").Input("x").Input("alpha").Output("y").Build());
  }
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

class HardSigmoidFunctor : public UnaryFunctor {
 public:
  HardSigmoidFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("hardsigmoid").Input("in").Output("out").Build());
  }
};

class SoftmaxFunctor : public UnaryFunctor {
 public:
  SoftmaxFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("softmax").Input("in").Output("out").Build());
  }
};

class HardSwishFunctor : public UnaryFunctor {
 public:
  HardSwishFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("hardswish").Input("in").Output("out").Build());
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

}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<impl::ReluFunctor>("Relu");
  m.add_functor<impl::PReluFunctor>("PRelu");
  m.add_functor<impl::HardTanhFunctor>("HardTanh");
  m.add_functor<impl::HardTanhGradFunctor>("HardTanhGrad");
  m.add_functor<impl::EluFunctor>("Elu");
  m.add_functor<impl::EluGradFunctor>("EluGrad");
  m.add_functor<impl::GeluFunctor>("Gelu");
  m.add_functor<impl::HardSigmoidFunctor>("HardSigmoid");
  m.add_functor<impl::SoftmaxFunctor>("Softmax");
  m.add_functor<impl::HardSwishFunctor>("HardSwish");
  m.add_functor<impl::LeakyReluFunctor>("LeakyRelu");
  m.add_functor<impl::LeakyReluGradFunctor>("LeakyReluGrad");
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow
