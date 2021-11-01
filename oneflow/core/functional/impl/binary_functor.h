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

#ifndef ONEFLOW_CORE_FUNCTIONAL_IMPL_BINARY_FUNCTOR_H_
#define ONEFLOW_CORE_FUNCTIONAL_IMPL_BINARY_FUNCTOR_H_

#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/functional/impl/common.h"
#include "oneflow/core/functional/tensor_processor.h"

namespace oneflow {
namespace one {
namespace functional {

namespace impl {

class BinaryFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& y) const {
    TensorProcessor tensor_processor;
    JUST(tensor_processor.PromoteInputsToCommonDtype(true).AddInputs({x, y}).Apply());
    TensorTuple input_tuple = JUST(tensor_processor.GetInputs());
    return OpInterpUtil::Dispatch<Tensor>(*op_, input_tuple);
  }

 protected:
  BinaryFunctor() = default;
  virtual ~BinaryFunctor() = default;

  std::shared_ptr<OpExpr> op_;
};

class BinaryFloatFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& y) const {
    TensorProcessor tensor_processor;
    JUST(tensor_processor.AddInputs({x, y}, DType::Float()).Apply());
    TensorTuple input_tuple = JUST(tensor_processor.GetInputs());
    return OpInterpUtil::Dispatch<Tensor>(*op_, input_tuple);
  }

 protected:
  BinaryFloatFunctor() = default;
  virtual ~BinaryFloatFunctor() = default;

  std::shared_ptr<OpExpr> op_;
};

class InplaceableBinaryFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& y, bool inplace) const {
    if (inplace) {
      JUST(CheckInplaceValid(x));
      JUST(CheckShapeCanExpandTo(*y->shape(), *x->shape()));
      std::shared_ptr<TensorTuple> outputs = std::make_shared<TensorTuple>(1);
      outputs->at(0) = x;
      JUST(OpInterpUtil::Dispatch(*op_, {x, y}, outputs.get()));
      return outputs->at(0);
    } else {
      return OpInterpUtil::Dispatch<Tensor>(*op_, {x, y});
    }
  }

 protected:
  InplaceableBinaryFunctor() = default;
  virtual ~InplaceableBinaryFunctor() = default;

  std::shared_ptr<OpExpr> op_;
};

}  // namespace impl

}  // namespace functional
}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FUNCTIONAL_IMPL_BINARY_FUNCTOR_H_
