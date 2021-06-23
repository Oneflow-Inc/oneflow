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

#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/functional/function_library.h"
#include "oneflow/core/functional/impl/common.h"
#include "oneflow/core/functional/impl/unary_functor.h"
#include "oneflow/core/functional/scalar.h"

namespace oneflow {
namespace one {
namespace functional {

namespace impl {

class BiasAddFunctor {
 public:
  BiasAddFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("bias_add").Input("a").Input("b").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& bias, const int32_t& axis) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("axis", axis));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x, bias}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class MatMulBaseFunctor {
 public:
  MatMulBaseFunctor() = default;
  virtual ~MatMulBaseFunctor() = default;
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& a,
                           const std::shared_ptr<one::Tensor>& b, const bool& transpose_a,
                           const bool& transpose_b, const double& alpha) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<bool>("transpose_a", transpose_a));
    JUST(attrs.SetAttr<bool>("transpose_b", transpose_b));
    JUST(attrs.SetAttr<double>("alpha", alpha));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {a, b}, attrs);
  }

 protected:
  std::shared_ptr<OpExpr> op_;
};

class MatMulFunctor : public MatMulBaseFunctor {
 public:
  MatMulFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("matmul").Input("a").Input("b").Output("out").Build());
  }
};

class BatchMatMulFunctor : public MatMulBaseFunctor {
 public:
  BatchMatMulFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("batch_matmul").Input("a").Input("b").Output("out").Build());
  }
};

class BroadcastMatMulFunctor : public MatMulBaseFunctor {
 public:
  BroadcastMatMulFunctor() {
    op_ =
        CHECK_JUST(one::OpBuilder("broadcast_matmul").Input("a").Input("b").Output("out").Build());
  }
};

class LayerNormFunctor {
 public:
  LayerNormFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("layer_norm")
                         .Input("x")
                         .Output("y")
                         .Output("mean")
                         .Output("inv_variance")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const int64_t& begin_norm_axis,
                           const int64_t& begin_params_axis, const double& epsilon) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("begin_norm_axis", begin_norm_axis));
    JUST(attrs.SetAttr<int64_t>("begin_params_axis", begin_params_axis));
    JUST(attrs.SetAttr<double>("epsilon", epsilon));
    JUST(attrs.SetAttr<bool>("center", false));
    JUST(attrs.SetAttr<bool>("scale", false));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class LayerNormAffineFunctor {
 public:
  LayerNormAffineFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("layer_norm")
                         .Input("x")
                         .Input("gamma")
                         .Input("beta")
                         .Output("y")
                         .Output("mean")
                         .Output("inv_variance")
                         .Output("normalized")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& gamma,
                           const std::shared_ptr<one::Tensor>& beta, const int64_t& begin_norm_axis,
                           const int64_t& begin_params_axis, const double& epsilon) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("begin_norm_axis", begin_norm_axis));
    JUST(attrs.SetAttr<int64_t>("begin_params_axis", begin_params_axis));
    JUST(attrs.SetAttr<double>("epsilon", epsilon));
    JUST(attrs.SetAttr<bool>("center", true));
    JUST(attrs.SetAttr<bool>("scale", true));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x, gamma, beta}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class SparseSoftmaxCrossEntropyFunctor {
 public:
  SparseSoftmaxCrossEntropyFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("sparse_softmax_cross_entropy")
                         .Input("prediction")
                         .Input("label")
                         .Output("out")
                         .Output("prob")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& logits,
                           const std::shared_ptr<one::Tensor>& label, const int64_t& depth) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("depth", depth));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {logits, label}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class NormalizationFunctor {
 public:
  NormalizationFunctor() {
    norm_eval_op_ = CHECK_JUST(one::OpBuilder("normalization")
                                   .Input("x")
                                   .Input("moving_mean")
                                   .Input("moving_variance")
                                   .Input("gamma")
                                   .Input("beta")
                                   .Output("y")
                                   .Attr("training", false)
                                   .Build());
    norm_training_op_ = CHECK_JUST(one::OpBuilder("normalization")
                                       .Input("x")
                                       .Input("moving_mean")
                                       .Input("moving_variance")
                                       .Input("gamma")
                                       .Input("beta")
                                       .Output("y")
                                       .Output("mean")
                                       .Output("inv_variance")
                                       .Attr("training", true)
                                       .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& moving_mean,
                           const std::shared_ptr<one::Tensor>& moving_variance,
                           const std::shared_ptr<one::Tensor>& gamma,
                           const std::shared_ptr<one::Tensor>& beta, const int32_t& axis,
                           const float& epsilon, const float& momentum,
                           const bool& is_training) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("axis", axis));
    JUST(attrs.SetAttr<float>("epsilon", epsilon));
    JUST(attrs.SetAttr<float>("momentum", momentum));
    std::shared_ptr<OpExpr> op;
    if (is_training) {
      op = norm_training_op_;
    } else {
      op = norm_eval_op_;
    }
    return OpInterpUtil::Dispatch<one::Tensor>(*op, {x, moving_mean, moving_variance, gamma, beta},
                                               attrs);
  }

 private:
  std::shared_ptr<OpExpr> norm_eval_op_;
  std::shared_ptr<OpExpr> norm_training_op_;
};

}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<impl::BiasAddFunctor>("BiasAdd");
  m.add_functor<impl::MatMulFunctor>("MatMul");
  m.add_functor<impl::BatchMatMulFunctor>("BatchMatMul");
  m.add_functor<impl::BroadcastMatMulFunctor>("BroadcastMatMul");
  m.add_functor<impl::LayerNormFunctor>("LayerNorm");
  m.add_functor<impl::LayerNormAffineFunctor>("LayerNormAffine");
  m.add_functor<impl::SparseSoftmaxCrossEntropyFunctor>("SparseSoftmaxCrossEntropy");
  m.add_functor<impl::NormalizationFunctor>("Normalization");
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow
