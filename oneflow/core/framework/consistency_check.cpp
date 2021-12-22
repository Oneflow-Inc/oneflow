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
#include <cstring>
#include "oneflow/core/framework/consistency_check.h"
#include "oneflow/core/job/rank_group.h"
#include "oneflow/core/framework/transport_util.h"
#include "oneflow/core/job/rank_group_scope.h"

namespace oneflow {

namespace {

template<typename Meta>
Maybe<void> MataConsistencyCheck(const Meta& meta) {
  const auto& meta_str = meta.DebugString();
  const char* meta_cstr = meta_str.c_str();
  size_t buffer_size = meta_str.length() + 1;
  std::vector<char> recv_buffer(buffer_size);
  char* recv_ptr = recv_buffer.data();
  const auto& rank_group = JUST(RankGroupScope::CurrentRankGroup());
  const auto& transport_token =
      JUST(TransportToken::NewTransportToken(kTransportTokenTypeCheckTensorConsistency));

  NaiveAsyncTransportCtx ctx(
      transport_token,
      [&](void** buffer, std::size_t* size, std::function<void()>* Cb) -> Maybe<void> {
        *buffer = const_cast<char*>(meta_cstr);
        *size = buffer_size;
        *Cb = [] {};
        return Maybe<void>::Ok();
      },
      [&](void** buffer, std::size_t* size, std::function<void()>* Cb) -> Maybe<void> {
        *buffer = recv_ptr;
        *size = buffer_size;
        *Cb = [] {};
        return Maybe<void>::Ok();
      });
  JUST(TransportUtil::SendToNextRankInRing(rank_group, transport_token, &ctx));
  JUST(TransportUtil::ReceiveFromPrevRankInRing(rank_group, transport_token, &ctx));
  JUST(TransportUtil::WaitUntilDoneOrTimeout(ctx, TransportUtil::TimeoutSeconds()));
  CHECK_OR_RETURN(std::memcmp(meta_cstr, reinterpret_cast<const void*>(recv_ptr), buffer_size)
                  == 0);
  return Maybe<void>::Ok();
}

template Maybe<void> MataConsistencyCheck<ParallelConf>(const ParallelConf&);
template Maybe<void> MataConsistencyCheck<cfg::NdSbp>(const cfg::NdSbp&);

}  // namespace

Maybe<void> DataConsistencyCheck(const void* buffer_ptr, size_t buffer_size,
                                 Symbol<ParallelDesc> placement) {
  const auto& rank_group = JUST(RankGroup::New(placement));

  std::vector<char> recv_buffer(buffer_size);
  char* recv_ptr = recv_buffer.data();

  TransportToken transport_token = JUST(TransportToken::NewTransportToken(kTransportTokenTypeData));
  NaiveAsyncTransportCtx ctx(
      transport_token,
      [&](void** buffer, std::size_t* size, std::function<void()>* Cb) -> Maybe<void> {
        *buffer = const_cast<void*>(buffer_ptr);
        *size = buffer_size;
        *Cb = [] {};
        return Maybe<void>::Ok();
      },
      [&](void** buffer, std::size_t* size, std::function<void()>* Cb) -> Maybe<void> {
        *buffer = recv_ptr;
        *size = buffer_size;
        *Cb = [] {};
        return Maybe<void>::Ok();
      });
  JUST(TransportUtil::SendToNextRankInRing(rank_group, transport_token, &ctx));
  JUST(TransportUtil::ReceiveFromPrevRankInRing(rank_group, transport_token, &ctx));
  JUST(TransportUtil::WaitUntilDoneOrTimeout(ctx, TransportUtil::TimeoutSeconds()));
  CHECK_OR_RETURN(std::memcmp(buffer_ptr, reinterpret_cast<const void*>(recv_ptr), buffer_size)
                  == 0)
      << "Each rank must have same input sequence or numpy array";
  return Maybe<void>::Ok();
}

Maybe<void> PlacementConsistencyCheck(Symbol<ParallelDesc> placement) {
  JUST(MataConsistencyCheck(placement->parallel_conf()));
  return Maybe<void>::Ok();
}

Maybe<void> NdSbpConsistencyCheck(Symbol<cfg::NdSbp> nd_sbp) {
  JUST(MataConsistencyCheck(*nd_sbp));
  return Maybe<void>::Ok();
}

Maybe<void> NdSbpConsistencyCheck(const std::vector<Symbol<cfg::SbpParallel>>& sbp_tuple) {
  const auto& nd_sbp = JUST(GetNdSbp(sbp_tuple));
  JUST(MataConsistencyCheck(*nd_sbp));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
