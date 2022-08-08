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
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_name_scope.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {
namespace one {

Maybe<Tensor> GradAccTryInsertUnpackAfterInput(const std::shared_ptr<Tensor>& input);
Maybe<Tensor> GradAccTryInsertRepeatAfterVar(const std::shared_ptr<Tensor>& variable);
Maybe<Tensor> GradAccTryInsertPackBeforeOutput(const std::shared_ptr<Tensor>& output);

Maybe<void> GradAccTryInsertRepeatTickBeforeSource(
    const std::shared_ptr<OperatorConf>& source_op_conf, bool is_local);

}  // namespace one
}  // namespace oneflow
