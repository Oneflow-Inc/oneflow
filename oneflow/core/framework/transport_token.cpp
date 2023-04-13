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
#include <array>
#include "oneflow/core/framework/transport_token.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/thread/thread_global_id.h"
#include "oneflow/core/framework/rank_group_rpc_util.h"

namespace oneflow {

/*static*/ Maybe<TransportToken> TransportToken::NewTransportToken(TransportTokenType type) {
  int32_t thread_global_id = GetThisThreadGlobalId();
  CHECK_GE_OR_RETURN(thread_global_id, 0);                             // NOLINT
  CHECK_LT_OR_RETURN(thread_global_id, MaxNumberOfThreadGlobalUId());  // NOLINT
  return TransportToken(type, thread_global_id);
}

Maybe<void> TransportToken::CheckThreadGlobalId() const {
  int32_t thread_global_id = GetThisThreadGlobalId();
  CHECK_EQ_OR_RETURN(thread_global_id, this->thread_global_id());  // NOLINT
  return Maybe<void>::Ok();
}

Maybe<void> TransportToken::set_src_rank(int64_t val) {
  CHECK_GE_OR_RETURN(val, 0);
  CHECK_LT_OR_RETURN(val, GetMaxVal<uint16_t>());
  src_rank_ = val;
  return Maybe<void>::Ok();
}

Maybe<void> TransportToken::set_dst_rank(int64_t val) {
  CHECK_GE_OR_RETURN(val, 0);
  CHECK_LT_OR_RETURN(val, GetMaxVal<uint16_t>());
  dst_rank_ = val;
  return Maybe<void>::Ok();
}

}  // namespace oneflow
