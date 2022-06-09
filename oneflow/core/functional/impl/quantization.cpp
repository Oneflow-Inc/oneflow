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

#include "oneflow/core/functional/impl/binary_functor.h"

#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/functional/function_library.h"

namespace oneflow {
namespace one {
namespace functional {

namespace impl {

class MinMaxObserverFunctor {
 public:
  MinMaxObserverFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("min_max_observer")
                         .Input("in")
                         .Output("scale")
                         .Output("zero_point")
                         .Build());
  }
  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& in,
                                const std::string& quantization_formula,
                                const int32_t& quantization_bit,
                                const std::string& quantization_scheme,
                                const bool& per_layer_quantization) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::string>("quantization_formula", quantization_formula));
    JUST(attrs.SetAttr<int32_t>("quantization_bit", quantization_bit));
    JUST(attrs.SetAttr<std::string>("quantization_scheme", quantization_scheme));
    JUST(attrs.SetAttr<bool>("per_layer_quantization", per_layer_quantization));
    return OpInterpUtil::Dispatch<TensorTuple>(*op_, {in}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class MovingAverageMinMaxObserverFunctor {
 public:
  MovingAverageMinMaxObserverFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("moving_average_min_max_observer")
                         .Input("in")
                         .Input("current_train_step")
                         .Input("moving_max")
                         .Input("moving_min")
                         .Output("scale")
                         .Output("zero_point")
                         .Build());
  }
  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& in,
                                const std::shared_ptr<one::Tensor>& current_train_step,
                                const std::shared_ptr<one::Tensor>& moving_max,
                                const std::shared_ptr<one::Tensor>& moving_min,
                                const bool& training, const int64_t& stop_update_after_iters,
                                const std::string& quantization_formula,
                                const int32_t& quantization_bit,
                                const std::string& quantization_scheme,
                                const float& momentum) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<bool>("training", training));
    JUST(attrs.SetAttr<std::string>("quantization_formula", quantization_formula));
    JUST(attrs.SetAttr<int64_t>("stop_update_after_iters", stop_update_after_iters));
    JUST(attrs.SetAttr<int32_t>("quantization_bit", quantization_bit));
    JUST(attrs.SetAttr<std::string>("quantization_scheme", quantization_scheme));
    JUST(attrs.SetAttr<float>("momentum", momentum));
    return OpInterpUtil::Dispatch<TensorTuple>(
        *op_, {in, current_train_step, moving_max, moving_min}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class FakeQuantizationFunctor {
 public:
  FakeQuantizationFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("fake_quantization")
                         .Input("in")
                         .Input("scale")
                         .Input("zero_point")
                         .Output("out")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& in,
                           const std::shared_ptr<one::Tensor>& scale,
                           const std::shared_ptr<one::Tensor>& zero_point,
                           const std::string& quantization_formula, const int32_t& quantization_bit,
                           const std::string& quantization_scheme) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::string>("quantization_formula", quantization_formula));
    JUST(attrs.SetAttr<int32_t>("quantization_bit", quantization_bit));
    JUST(attrs.SetAttr<std::string>("quantization_scheme", quantization_scheme));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {in, scale, zero_point}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class QuantizationFunctor {
 public:
  QuantizationFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("quantization")
                         .Input("in")
                         .Input("scale")
                         .Input("zero_point")
                         .Output("out")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& in,
                           const std::shared_ptr<one::Tensor>& scale,
                           const std::shared_ptr<one::Tensor>& zero_point,
                           const std::string quantization_formula, const int32_t& quantization_bit,
                           const std::string quantization_scheme) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::string>("quantization_formula", quantization_formula));
    JUST(attrs.SetAttr<int32_t>("quantization_bit", quantization_bit));
    JUST(attrs.SetAttr<std::string>("quantization_scheme", quantization_scheme));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {in, scale, zero_point}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) { m.add_functor<impl::FakeQuantizationFunctor>("FakeQuantization"); };
ONEFLOW_FUNCTION_LIBRARY(m) { m.add_functor<impl::QuantizationFunctor>("Quantization"); };
ONEFLOW_FUNCTION_LIBRARY(m) { m.add_functor<impl::MinMaxObserverFunctor>("MinMaxObserver"); };
ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<impl::MovingAverageMinMaxObserverFunctor>("MovingAverageMinMaxObserver");
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow
