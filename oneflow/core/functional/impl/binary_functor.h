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
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {
namespace functional {

namespace impl {

std::string TensorDeviceToString(const std::shared_ptr<Tensor>& tensor);

Maybe<void> CastDeviceForCPUScalarTensor(std::shared_ptr<Tensor>& tensor,
                                         std::shared_ptr<Tensor>& other, bool inplace);

class BinaryFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& y) const {
    auto tensor_x = x;
    auto tensor_y = y;
    JUST(CastDeviceForCPUScalarTensor(tensor_x, tensor_y, /*inplace=*/false));
    TensorProcessor tensor_processor;
    JUST(tensor_processor.PromoteInputsToCommonDtype(true).AddInputs({tensor_x, tensor_y}).Apply());
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
    auto tensor_x = x;
    auto tensor_y = y;
    JUST(CastDeviceForCPUScalarTensor(tensor_x, tensor_y, /*inplace=*/false));
    TensorProcessor tensor_processor;
    if (promoteTypes(tensor_x->dtype(), tensor_y->dtype())->is_integer()) {
      tensor_processor.AddInputs({tensor_x, tensor_y}, DType::Float());
    } else {
      tensor_processor.AddInputs({tensor_x, tensor_y})
          .PromoteInputsToCommonDtype(true)
          .PromoteIntegerInputsToFloatDtype(true);
    }
    JUST(tensor_processor.Apply());
    TensorTuple input_tuple = JUST(tensor_processor.GetInputs());
    return OpInterpUtil::Dispatch<Tensor>(*op_, input_tuple);
  }

 protected:
  BinaryFloatFunctor() = default;
  virtual ~BinaryFloatFunctor() = default;

  std::shared_ptr<OpExpr> op_;
};

class BinaryGradFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& y,
                           const std::shared_ptr<one::Tensor>& dz) const {
    TensorProcessor tensor_processor;
    JUST(tensor_processor.PromoteInputsToCommonDtype(true).AddInputs({x, y, dz}).Apply());
    TensorTuple input_tuple = JUST(tensor_processor.GetInputs());
    return OpInterpUtil::Dispatch<Tensor>(*op_, input_tuple);
  }

 protected:
  BinaryGradFunctor() = default;
  virtual ~BinaryGradFunctor() = default;

  std::shared_ptr<OpExpr> op_;
};

class InplaceableBinaryFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& y, bool inplace) const {
    auto tensor_x = x;
    auto tensor_y = y;
    JUST(CastDeviceForCPUScalarTensor(tensor_x, tensor_y, inplace));
    TensorProcessor tensor_processor;
    JUST(tensor_processor.PromoteInputsToCommonDtype(true).AddInputs({tensor_x, tensor_y}).Apply());
    TensorTuple input_tuple = JUST(tensor_processor.GetInputs());
    if (inplace) {
      std::shared_ptr<one::Tensor>& x_cast = input_tuple.at(0);
      std::shared_ptr<one::Tensor>& y_cast = input_tuple.at(1);
      JUST(CheckInplaceCastValid(x, x_cast));
      JUST(CheckInplaceShapeCanExpandTo(*y_cast->shape(), *x_cast->shape()));
      std::shared_ptr<TensorTuple> outputs = std::make_shared<TensorTuple>(1);
      outputs->at(0) = x_cast;
      JUST(OpInterpUtil::Dispatch(*op_, input_tuple, outputs.get()));
      return outputs->at(0);
    } else {
      return OpInterpUtil::Dispatch<Tensor>(*op_, input_tuple);
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
