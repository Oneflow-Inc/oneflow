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

#ifndef ONEFLOW_CORE_FUNCTIONAL_IMPL_UNARY_FUNCTOR_H_
#define ONEFLOW_CORE_FUNCTIONAL_IMPL_UNARY_FUNCTOR_H_

#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/functional/impl/common.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/functional/tensor_processor.h"

namespace oneflow {
namespace one {
namespace functional {

namespace impl {

class UnaryFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x) const {
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x});
  }

 protected:
  UnaryFunctor() = default;
  virtual ~UnaryFunctor() = default;

  std::shared_ptr<OpExpr> op_;
};

class InplaceUnaryFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x) const {
    JUST(CheckInplaceValid(x));
    std::shared_ptr<TensorTuple> outputs = std::make_shared<TensorTuple>(1);
    outputs->at(0) = x;
    if (x->requires_grad()) {
      JUST(OpInterpUtil::Dispatch(*op_, {JUST(functional::Identity(x))}, outputs.get()));
    } else {
      JUST(OpInterpUtil::Dispatch(*op_, {x}, outputs.get()));
    }
    return outputs->at(0);
  }

 protected:
  InplaceUnaryFunctor() = default;
  virtual ~InplaceUnaryFunctor() = default;

  std::shared_ptr<OpExpr> op_;
};

class FloatUnaryFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x) const {
    // The functor lowest Dtype is Float32. (For sigmoid, tanh and etc. )
    TensorProcessor tensor_processor;
    JUST(tensor_processor.AddInputs({x}, DType::Float())
             .PromoteInputsToCommonDtype(true)
             .PromoteIntegerInputsToFloatDtype(true)
             .Apply());
    TensorTuple input_tuple = JUST(tensor_processor.GetInputs());
    return OpInterpUtil::Dispatch<one::Tensor>(*op_, input_tuple);
  }

 protected:
  FloatUnaryFunctor() = default;
  virtual ~FloatUnaryFunctor() = default;

  std::shared_ptr<OpExpr> op_;
};

class InplaceFloatUnaryFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x) const {
    TensorProcessor tensor_processor;
    JUST(tensor_processor.AddInputs({x}, DType::Float()).Apply());
    TensorTuple input_tuple = JUST(tensor_processor.GetInputs());
    JUST(CheckInplaceCastValid(x, input_tuple.at(0)));
    JUST(CheckInplaceValid(x));
    std::shared_ptr<TensorTuple> outputs = std::make_shared<TensorTuple>(1);
    outputs->at(0) = x;
    if (x->requires_grad()) {
      // It should copy input tensor in autograd_mode because these operators can't calculate
      // in_grad with output.
      JUST(OpInterpUtil::Dispatch(*op_, {JUST(functional::Identity(x))}, outputs.get()));
    } else {
      JUST(OpInterpUtil::Dispatch(*op_, {x}, outputs.get()));
    }
    return outputs->at(0);
  }

 protected:
  InplaceFloatUnaryFunctor() = default;
  virtual ~InplaceFloatUnaryFunctor() = default;

  std::shared_ptr<OpExpr> op_;
};

}  // namespace impl

}  // namespace functional
}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FUNCTIONAL_IMPL_UNARY_FUNCTOR_H_
