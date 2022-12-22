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
#include "oneflow/core/common/decorator.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/framework/transport_token.h"
#include "oneflow/core/framework/tensor_global_id.h"

namespace oneflow {

namespace {

Maybe<std::shared_ptr<TransportToken>> RawGetMetaTransportToken() {
  const auto& token = JUST(TransportToken::NewTransportToken(kTransportTokenTypeMeta));
  return std::make_shared<TransportToken>(token);
}
static constexpr auto* GetMetaTransportToken = DECORATE(&RawGetMetaTransportToken, ThreadLocal);

}  // namespace

Maybe<TransportToken> NewTensorGlobalId() { return ++**JUST(GetMetaTransportToken()); }

namespace one {

int64_t* MutThreadLocalGlobalIdDepth() {
  static thread_local int64_t recursive_depth = 0;
  return &recursive_depth;
}

Maybe<void> InitGlobalId(TensorTuple* outputs) {
  for (const auto& output : *outputs) {
    CHECK_OR_RETURN(output);
    const auto& global_tensor = JUST(output->AsGlobalTensor());
    CHECK_OR_RETURN(global_tensor)
        << Error::UnimplementedError() << "global tensors suppported only.";
    const auto& transport_token = JUST(NewTensorGlobalId());
    JUST(global_tensor->mut_impl()->set_transport_token(transport_token));
  }
  return Maybe<void>::Ok();
}

}  // namespace one

}  // namespace oneflow
