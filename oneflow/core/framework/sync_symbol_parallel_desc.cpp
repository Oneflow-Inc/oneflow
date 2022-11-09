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
#include "oneflow/core/framework/sync_symbol_parallel_desc.h"
#include "oneflow/core/framework/rank_group_rpc_util.h"
#include "oneflow/core/job/rank_group_scope.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/common/constant.h"

namespace oneflow {

namespace {

static const int kLimitParallelConfString = 1024 * 64;
struct FlatParallelConf {
  size_t available_size() const {
    CHECK_GE(this->buffer_size, 0) << "Buffer size should be non-negative";
    CHECK_LT(this->buffer_size, kLimitParallelConfString)
        << "Buffer size should be less than " << kLimitParallelConfString;
    return sizeof(FlatParallelConf) - kLimitParallelConfString + this->buffer_size;
  }

  size_t capacity() const { return sizeof(FlatParallelConf); }

  static Maybe<FlatParallelConf> New(uint64_t symbol_id, Symbol<ParallelDesc> parallel_desc) {
    const auto& data = std::make_shared<FlatParallelConf>();
    JUST(data->Init(symbol_id, parallel_desc));
    return data;
  }

  Maybe<void> Init(uint64_t symbol_id, Symbol<ParallelDesc> parallel_desc) {
    const auto& parallel_conf = parallel_desc->parallel_conf();
    int64_t byte_size = parallel_conf.ByteSize();
    CHECK_LE_OR_RETURN(byte_size, kLimitParallelConfString)
        << Error::InvalidValueError() << "Byte size of parallel description should be less than "
        << kLimitParallelConfString << ", but got " << byte_size;
    this->symbol_id = symbol_id;
    this->buffer_size = byte_size;
    CHECK_OR_RETURN(parallel_conf.SerializeToArray(this->buffer, kLimitParallelConfString))
        << Error::RuntimeError()
        << "Error serializing parallel description: " << parallel_conf.ShortDebugString();
    return Maybe<void>::Ok();
  }

  Maybe<void> Check(uint64_t symbol_id, Symbol<ParallelDesc> parallel_desc) const {
    const auto& parallel_conf = parallel_desc->parallel_conf();
    int64_t byte_size = parallel_conf.ByteSize();
    const auto& debugString = parallel_conf.ShortDebugString();
    CHECK_LE_OR_RETURN(byte_size, kLimitParallelConfString)
        << Error::InvalidValueError() << "Byte size of parallel description should be less than "
        << kLimitParallelConfString << ", but got " << byte_size;
    CHECK_EQ_OR_RETURN(this->symbol_id, symbol_id) << Error::RuntimeError() << "expected symbol id "
                                                   << symbol_id << ", but got " << this->symbol_id;
    CHECK_EQ_OR_RETURN(this->buffer_size, byte_size)
        << Error::RuntimeError() << "Inconsistent parallel description: " << debugString;
    std::vector<char> serialized(byte_size);
    CHECK_OR_RETURN(parallel_conf.SerializeToArray(serialized.data(), kLimitParallelConfString))
        << Error::RuntimeError() << "Error serializing parallel description: " << debugString;
    CHECK_EQ_OR_RETURN(std::memcmp(serialized.data(), this->buffer, byte_size), 0)
        << Error::RuntimeError() << "Inconsistent parallel description: " << debugString;
    return Maybe<void>::Ok();
  }

  uint64_t symbol_id;
  uint64_t buffer_size;
  char buffer[kLimitParallelConfString];
};

}  // namespace

Maybe<void> SyncSymbolParallelDesc(uint64_t symbol_id, Symbol<ParallelDesc> parallel_desc) {
  const auto& transport_token =
      JUST(TransportToken::NewTransportToken(kTransportTokenTypeSyncSymbolParallelDesc));
  const auto& recv_buffer = std::make_shared<FlatParallelConf>();
  NaiveAsyncTransportCtx ctx(
      transport_token,
      [&](void** buffer, std::size_t* size, std::function<void()>* Cb) -> Maybe<void> {
        const auto& send_buffer = JUST(FlatParallelConf::New(symbol_id, parallel_desc));
        *buffer = send_buffer.get();
        *size = send_buffer->available_size();
        *Cb = [send_buffer] {};
        return Maybe<void>::Ok();
      },
      [recv_buffer](void** buffer, std::size_t* size, std::function<void()>* Cb) -> Maybe<void> {
        *buffer = recv_buffer.get();
        *size = recv_buffer->capacity();
        *Cb = [recv_buffer] {};
        return Maybe<void>::Ok();
      });
  const auto& rank_group = JUST(RankGroupScope::CurrentRankGroup());
  JUST(TransportUtil::SendToNextRankInRing(rank_group, transport_token, &ctx));
  JUST(TransportUtil::ReceiveFromPrevRankInRing(rank_group, transport_token, &ctx));
  JUST_MSG(ctx.WaitDone(), kAsymmetricCodeErrorMsg);
  JUST(recv_buffer->Check(symbol_id, parallel_desc));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
