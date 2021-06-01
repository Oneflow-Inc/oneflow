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

class Conv2DFunctor {
 public:
  Conv2DFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("conv2d").Input("in").Input("weight").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& weight, const int32_t& filters,
                           const std::vector<int32_t>& kernel_size,
                           const std::vector<int32_t>& strides,
                           const std::vector<int32_t>& padding_before,
                           const std::vector<int32_t>& dilation_rate, const int32_t& groups,
                           const std::string& data_format) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("filters", filters));
    JUST(attrs.SetAttr<std::vector<int32_t>>("kernel_size", kernel_size));
    JUST(attrs.SetAttr<std::vector<int32_t>>("strides", strides));
    JUST(attrs.SetAttr<std::vector<int32_t>>("padding_before", padding_before));
    JUST(attrs.SetAttr<std::vector<int32_t>>("dilation_rate", dilation_rate));
    JUST(attrs.SetAttr<int32_t>("groups", groups));
    JUST(attrs.SetAttr<std::string>("data_format", data_format));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x, weight}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class MatMulFunctor {
 public:
  MatMulFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("matmul").Input("a").Input("b").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& a,
                           const std::shared_ptr<one::Tensor>& b, const bool& transpose_a,
                           const bool& transpose_b, const double& alpha) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<bool>("transpose_a", transpose_a));
    JUST(attrs.SetAttr<bool>("transpose_b", transpose_b));
    JUST(attrs.SetAttr<double>("alpha", alpha));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {a, b}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class BroadcastMatMulFunctor {
 public:
  BroadcastMatMulFunctor() {
    op_ =
        CHECK_JUST(one::OpBuilder("broadcast_matmul").Input("a").Input("b").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& a,
                           const std::shared_ptr<one::Tensor>& b, const bool& transpose_a,
                           const bool& transpose_b, const double& alpha) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<bool>("transpose_a", transpose_a));
    JUST(attrs.SetAttr<bool>("transpose_b", transpose_b));
    JUST(attrs.SetAttr<double>("alpha", alpha));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {a, b}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<impl::BiasAddFunctor>("BiasAdd");
  m.add_functor<impl::Conv2DFunctor>("Conv2D");
  m.add_functor<impl::MatMulFunctor>("MatMul");
  m.add_functor<impl::BroadcastMatMulFunctor>("BroadcastMatMul");
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow
