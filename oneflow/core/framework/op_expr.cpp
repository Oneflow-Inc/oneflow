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
#include "oneflow/core/framework/op_expr.h"

namespace oneflow {
namespace one {

std::shared_ptr<OpExpr> UserOpExpr::GetBackwardOpExpr() const {
  // TODO(hjchen2)
  return std::shared_ptr<OpExpr>(new UserOpExpr);
}

std::shared_ptr<OpExpr> VariableOpExpr::GetBackwardOpExpr() const {
  // TODO(hjchen2)
  return std::shared_ptr<OpExpr>(new VariableOpExpr);
}

std::shared_ptr<OpExpr> CastToMirroredOpExpr::GetBackwardOpExpr() const {
  // TODO(hjchen2)
  return std::shared_ptr<OpExpr>(new CastToMirroredOpExpr);
}

std::shared_ptr<OpExpr> CastFromMirroredOpExpr::GetBackwardOpExpr() const {
  // TODO(hjchen2)
  return std::shared_ptr<OpExpr>(new CastFromMirroredOpExpr);
}

std::shared_ptr<OpExpr> DistributeSplitOpExpr::GetBackwardOpExpr() const {
  // TODO(hjchen2)
  return std::shared_ptr<OpExpr>(new DistributeSplitOpExpr);
}

std::shared_ptr<OpExpr> DistributeCloneOpExpr::GetBackwardOpExpr() const {
  // TODO(hjchen2)
  return std::shared_ptr<OpExpr>(new DistributeCloneOpExpr);
}

std::shared_ptr<OpExpr> DistributeConcatOpExpr::GetBackwardOpExpr() const {
  // TODO(hjchen2)
  return std::shared_ptr<OpExpr>(new DistributeConcatOpExpr);
}

std::shared_ptr<OpExpr> DistributeAddOpExpr::GetBackwardOpExpr() const {
  // TODO(hjchen2)
  return std::shared_ptr<OpExpr>(new DistributeAddOpExpr);
}

std::shared_ptr<OpExpr> FunctionOpExpr::GetBackwardOpExpr() const {
  // TODO(hjchen2)
  return std::shared_ptr<OpExpr>(new FunctionOpExpr);
}

}  // namespace one
}  // namespace oneflow
