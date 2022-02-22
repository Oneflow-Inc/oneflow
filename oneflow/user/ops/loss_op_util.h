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
#ifndef ONEFLOW_USER_OPS_LOSS_OP_UTIL_H_
#define ONEFLOW_USER_OPS_LOSS_OP_UTIL_H_

#include <functional>
#include "oneflow/core/framework/framework.h"

namespace oneflow {

user_op::GetSbpFn GenLossForwardDefaultGetSbpFn(
    const std::function<void(user_op::UserOpSbpSignatureBuilder& builder,
                             user_op::SbpContext* ctx)>& f =
        [](user_op::UserOpSbpSignatureBuilder& builder, user_op::SbpContext* ctx) {});

user_op::GetSbpFn GenLossBackwardDefaultGetSbpFn(
    const std::function<void(user_op::UserOpSbpSignatureBuilder& builder,
                             user_op::SbpContext* ctx)>& f =
        [](user_op::UserOpSbpSignatureBuilder& builder, user_op::SbpContext* ctx) {});

}  // namespace oneflow

#endif  // ONEFLOW_USER_OPS_LOSS_OP_UTIL_H_
