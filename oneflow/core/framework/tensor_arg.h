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

#ifndef ONEFLOW_CORE_FRAMEWORK_TENSOR_ARG_H_
#define ONEFLOW_CORE_FRAMEWORK_TENSOR_ARG_H_

#include <memory>
#include <vector>
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace one {

class Tensor;
class OpExpr;

// This class will be used in TensorImpl and Autograd. It will share data with different
// FunctionNodes.
class TensorArg final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TensorArg);
  TensorArg();
  ~TensorArg() = default;

  bool Empty() const;
  void Release();
  Maybe<void> PushPartialTensor(const std::shared_ptr<Tensor>& partial_tensor);
  Maybe<Tensor> GetAccTensor();

 private:
  std::shared_ptr<Tensor> acc_tensor_;
  std::shared_ptr<OpExpr> add2_op_;
};

}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_TENSOR_ARG_H_
