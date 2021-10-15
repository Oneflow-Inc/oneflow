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
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/data_consistency_check.h"
#include "oneflow/core/job/rank_group.h"
#include "oneflow/core/framework/transport_util.h"

namespace oneflow {

namespace {

template<typename T>
bool CheckVecEqual(size_t size, const T* in0, const T* in1) {
  for (size_t i = 0; i < size; ++i) {
    if (*(in0 + i) != *(in1 + i)) { return false; }
  }
  return true;
}

}  // namespace

template<typename T>
Maybe<void> DataConsistencyCheck(const void* buffer_ptr, size_t elem_cnt,
                                 Symbol<ParallelDesc> placement) {
  const auto& rank_group = JUST(RankGroup::New(placement));
  size_t data_size = elem_cnt * sizeof(T);

  std::vector<T> recv_buffer(elem_cnt);
  T* recv_ptr = recv_buffer.data();

  TransportToken transport_token = JUST(TransportToken::NewTransportToken(kTransportTokenTypeData));
  NaiveAsyncTransportCtx ctx(
      transport_token,
      [&](void** buffer, std::size_t* size, std::function<void()>* Cb) -> Maybe<void> {
        *buffer = const_cast<void*>(buffer_ptr);
        *size = data_size;
        *Cb = [] {};
        return Maybe<void>::Ok();
      },
      [&](void** buffer, std::size_t* size, std::function<void()>* Cb) -> Maybe<void> {
        *buffer = recv_ptr;
        *size = data_size;
        *Cb = [] {};
        return Maybe<void>::Ok();
      });
  JUST(TransportUtil::SendToNextRankInRing(rank_group, transport_token, &ctx));
  JUST(TransportUtil::ReceiveFromPrevRankInRing(rank_group, transport_token, &ctx));
  JUST(TransportUtil::WaitUntilDoneOrTimeout(ctx, TransportUtil::TimeoutSeconds()));
  CHECK_OR_RETURN(CheckVecEqual(elem_cnt, reinterpret_cast<const T*>(buffer_ptr), recv_ptr))
      << "Each rank must have same input sequence or numpy array";
  return Maybe<void>::Ok();
}

#define INSTATIATION_DATA_CONSISTENCY_CHECK(type_cpp, type_proto)                              \
  template Maybe<void> DataConsistencyCheck<type_cpp>(const void* buffer_ptr, size_t elem_cnt, \
                                                      Symbol<ParallelDesc> placement);

OF_PP_FOR_EACH_TUPLE(INSTATIATION_DATA_CONSISTENCY_CHECK, POD_DATA_TYPE_SEQ)
#undef INSTATIATION_DATA_CONSISTENCY_CHECK

}  // namespace oneflow
