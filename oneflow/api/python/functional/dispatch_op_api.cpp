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

#include "oneflow/core/common/scalar.h"
#include "oneflow/core/framework/op_interp_ctx.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/functional/function_library.h"

namespace oneflow {
namespace one {
namespace functional {

namespace impl {

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor(
      "DispatchFeedInput",
      [](const std::shared_ptr<OpExpr>& op, const std::shared_ptr<Tensor>& input) -> Maybe<Tensor> {
        auto ctx = std::make_shared<FeedInputOpInterpCtx>();
        return OpInterpUtil::Dispatch<Tensor>(*op, {input}, ctx);
      });
  m.add_functor(
      "DispatchFetchOutput",
      [](const std::shared_ptr<OpExpr>& op, const std::shared_ptr<Tensor>& input) -> Maybe<Tensor> {
        auto ctx = std::make_shared<FetchOutputOpInterpCtx>();
        return OpInterpUtil::Dispatch<Tensor>(*op, {input}, ctx);
      });
  m.add_functor("DispatchFeedVariable",
                [](const std::shared_ptr<OpExpr>& op, const std::shared_ptr<Tensor>& input,
                   const Scalar& l2) -> Maybe<Tensor> {
                  auto ctx = std::make_shared<FeedVariableOpInterpCtx>();
                  ctx->l2 = JUST(l2.As<double>());
                  return OpInterpUtil::Dispatch<Tensor>(*op, {input}, ctx);
                });
}

}  // namespace impl

}  // namespace functional
}  // namespace one
}  // namespace oneflow
