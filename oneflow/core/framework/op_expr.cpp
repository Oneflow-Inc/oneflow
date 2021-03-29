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
#include "oneflow/core/framework/op_expr_grad_function.h"

namespace oneflow {
namespace one {

Maybe<OpExprGradClosure> UserOpExpr::GetOrCreateOpGradClosure() const {
  if (!op_grad_closure_.get()) {
    if (IsClassRegistered<std::string, OpExprGradFunction>(proto().op_type_name())) {
      op_grad_closure_.reset(NewObj<std::string, OpExprGradFunction>(proto().op_type_name()));
    } else {
      op_grad_closure_.reset(NewObj<std::string, OpExprGradFunction>("default"));
    }
  }
  CHECK_NOTNULL_OR_RETURN(op_grad_closure_.get());
  op_grad_closure_->Init(*this);
  return std::make_shared<OpExprGradClosure>(op_grad_closure_);
}

Maybe<OpExprGradClosure> VariableOpExpr::GetOrCreateOpGradClosure() const { UNIMPLEMENTED(); }

Maybe<OpExprGradClosure> CastToMirroredOpExpr::GetOrCreateOpGradClosure() const { UNIMPLEMENTED(); }

Maybe<OpExprGradClosure> CastFromMirroredOpExpr::GetOrCreateOpGradClosure() const {
  UNIMPLEMENTED();
}

Maybe<OpExprGradClosure> DistributeSplitOpExpr::GetOrCreateOpGradClosure() const {
  UNIMPLEMENTED();
}

Maybe<OpExprGradClosure> DistributeCloneOpExpr::GetOrCreateOpGradClosure() const {
  UNIMPLEMENTED();
}

Maybe<OpExprGradClosure> DistributeConcatOpExpr::GetOrCreateOpGradClosure() const {
  UNIMPLEMENTED();
}

Maybe<OpExprGradClosure> DistributeAddOpExpr::GetOrCreateOpGradClosure() const { UNIMPLEMENTED(); }

}  // namespace one
}  // namespace oneflow
