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
#include "oneflow/core/framework/op_expr_grad_closure.h"

namespace oneflow {
namespace one {

Maybe<OpExprGradClosureWrapper> UserOpExpr::GetOrCreateOpGradClosure() const {
  if (!op_grad_closure_.get()) {
    if (IsClassRegistered<std::string, OpExprGradClosure>(proto().op_type_name())) {
      op_grad_closure_.reset(NewObj<std::string, OpExprGradClosure>(proto().op_type_name()));
    } else {
      op_grad_closure_.reset(NewObj<std::string, OpExprGradClosure>("default"));
    }
  }
  CHECK_NOTNULL_OR_RETURN(op_grad_closure_.get());
  op_grad_closure_->Init(*this);
  return std::make_shared<OpExprGradClosureWrapper>(op_grad_closure_);
}

Maybe<OpExprGradClosureWrapper> VariableOpExpr::GetOrCreateOpGradClosure() const {
  UNIMPLEMENTED();
}

Maybe<OpExprGradClosureWrapper> CastToMirroredOpExpr::GetOrCreateOpGradClosure() const {
  UNIMPLEMENTED();
}

Maybe<OpExprGradClosureWrapper> CastFromMirroredOpExpr::GetOrCreateOpGradClosure() const {
  UNIMPLEMENTED();
}

Maybe<OpExprGradClosureWrapper> DistributeSplitOpExpr::GetOrCreateOpGradClosure() const {
  UNIMPLEMENTED();
}

Maybe<OpExprGradClosureWrapper> DistributeCloneOpExpr::GetOrCreateOpGradClosure() const {
  UNIMPLEMENTED();
}

Maybe<OpExprGradClosureWrapper> DistributeConcatOpExpr::GetOrCreateOpGradClosure() const {
  UNIMPLEMENTED();
}

Maybe<OpExprGradClosureWrapper> DistributeAddOpExpr::GetOrCreateOpGradClosure() const {
  UNIMPLEMENTED();
}

}  // namespace one
}  // namespace oneflow
