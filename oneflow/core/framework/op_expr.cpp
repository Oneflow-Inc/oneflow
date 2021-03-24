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

#include "oneflow/core/common/auto_registration_factory.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_expr_grad.h"

namespace oneflow {
namespace one {

Maybe<OpExprGradInterface> UserOpExpr::GetOrCreateOpGrad() const {
  if (!op_grad_.get()) { op_grad_.reset(NewObj<std::string, OpExprGrad>(type())); }
  op_grad_->Init(*this);
  return std::make_shared<OpExprGradInterface>(op_grad_);
}

Maybe<OpExprGradInterface> VariableOpExpr::GetOrCreateOpGrad() const { UNIMPLEMENTED(); }

Maybe<OpExprGradInterface> CastToMirroredOpExpr::GetOrCreateOpGrad() const { UNIMPLEMENTED(); }

Maybe<OpExprGradInterface> CastFromMirroredOpExpr::GetOrCreateOpGrad() const { UNIMPLEMENTED(); }

Maybe<OpExprGradInterface> DistributeSplitOpExpr::GetOrCreateOpGrad() const { UNIMPLEMENTED(); }

Maybe<OpExprGradInterface> DistributeCloneOpExpr::GetOrCreateOpGrad() const { UNIMPLEMENTED(); }

Maybe<OpExprGradInterface> DistributeConcatOpExpr::GetOrCreateOpGrad() const { UNIMPLEMENTED(); }

Maybe<OpExprGradInterface> DistributeAddOpExpr::GetOrCreateOpGrad() const { UNIMPLEMENTED(); }

}  // namespace one
}  // namespace oneflow
