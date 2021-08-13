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
#include "oneflow/core/thread/consistent_unique_id.h"
#include "oneflow/core/framework/rank_group_rpc_util.h"

namespace oneflow {

namespace {

class DataTransportTokenView final {
 public:
  static Maybe<DataTransportTokenView*> MutCast(TransportToken* transport_token) {
    CHECK_EQ_OR_RETURN(transport_token->type(), kDataTransportTokenType);
    return reinterpret_cast<DataTransportTokenView*>(transport_token);
  }

  void set_data_seq_id(int64_t seq_id) { data_seq_id_ = seq_id; }

 private:
  uint16_t src_rank_;
  uint16_t dst_rank_;
  uint32_t type_ : 2;  // TransportTokenType
  uint32_t data_seq_id_ : 30;
};
static_assert(sizeof(DataTransportTokenView) == sizeof(uint64_t), "");

class MetaTransportTokenView final {
 public:
  int64_t thread_consistent_unique_id() const { return thread_consistent_unique_id_; }
  int64_t rank_group_level() const { return rank_group_level_; }

  static Maybe<MetaTransportTokenView*> MutCast(TransportToken* transport_token) {
    CHECK_EQ_OR_RETURN(transport_token->type(), kMetaTransportTokenType);
    return reinterpret_cast<MetaTransportTokenView*>(transport_token);
  }

  static Maybe<const MetaTransportTokenView*> Cast(const TransportToken* transport_token) {
    CHECK_EQ_OR_RETURN(transport_token->type(), kMetaTransportTokenType);
    return reinterpret_cast<const MetaTransportTokenView*>(transport_token);
  }

  Maybe<void> set_thread_consistent_unique_id(int8_t val) {
    CHECK_GE_OR_RETURN(val, 0);
    CHECK_LT_OR_RETURN(val, 1 << kTransportTokenThreadConsistentUIdBit);
    thread_consistent_unique_id_ = val;
    return Maybe<void>::Ok();
  }

  Maybe<void> set_rank_group_level(int32_t val) {
    CHECK_GE_OR_RETURN(val, 0);
    CHECK_LT_OR_RETURN(val, 1 << kTransportTokenRankGroupLevelBit);
    rank_group_level_ = val;
    return Maybe<void>::Ok();
  }

  MetaTransportTokenView& operator++() {
    ++low_meta_seq_id_;
    if (low_meta_seq_id_ == 0) { ++high_meta_seq_id_; }
    return *this;
  }

 private:
  uint16_t src_rank_;
  uint16_t dst_rank_;
  uint8_t type_ : 2;  // TransportTokenType
  uint8_t thread_consistent_unique_id_ : kTransportTokenThreadConsistentUIdBit;
  uint8_t rank_group_level_ : kTransportTokenRankGroupLevelBit;
  uint8_t high_meta_seq_id_;
  uint16_t low_meta_seq_id_;
};
static_assert(sizeof(MetaTransportTokenView) == sizeof(uint64_t), "");

class CtrlTransportTokenView final {
 public:
  int64_t thread_consistent_unique_id() const { return thread_consistent_unique_id_; }
  int64_t rank_group_level() const { return rank_group_level_; }

  static Maybe<CtrlTransportTokenView*> MutCast(TransportToken* transport_token) {
    CHECK_EQ_OR_RETURN(transport_token->type(), kCtrlTransportTokenType);
    return reinterpret_cast<CtrlTransportTokenView*>(transport_token);
  }

  static Maybe<const CtrlTransportTokenView*> Cast(const TransportToken* transport_token) {
    CHECK_EQ_OR_RETURN(transport_token->type(), kCtrlTransportTokenType);
    return reinterpret_cast<const CtrlTransportTokenView*>(transport_token);
  }

  Maybe<void> set_thread_consistent_unique_id(int8_t val) {
    CHECK_GE_OR_RETURN(val, 0);
    CHECK_LT_OR_RETURN(val, 1 << kTransportTokenThreadConsistentUIdBit);
    thread_consistent_unique_id_ = val;
    return Maybe<void>::Ok();
  }
  Maybe<void> set_rank_group_level(int32_t val) {
    CHECK_GE_OR_RETURN(val, 0);
    CHECK_LT_OR_RETURN(val, 1 << kTransportTokenRankGroupLevelBit);
    rank_group_level_ = val;
    return Maybe<void>::Ok();
  }

  RankGroupCtrlCmd cmd() const { return static_cast<RankGroupCtrlCmd>(cmd_); }

  void set_cmd(RankGroupCtrlCmd cmd) {
    static_assert(kSizeOfRankGroupCtrlCmd < (1 << 8), "");
    cmd_ = static_cast<int8_t>(cmd);
  }

  void set_ctrl_seq_id(int32_t val) { ctrl_seq_id_ = val; }

 private:
  uint16_t src_rank_;
  uint16_t dst_rank_;
  uint8_t type_ : 2;  // TransportTokenType
  uint8_t thread_consistent_unique_id_ : kTransportTokenThreadConsistentUIdBit;
  uint8_t rank_group_level_ : kTransportTokenRankGroupLevelBit;
  uint8_t cmd_;
  uint16_t ctrl_seq_id_;
};
static_assert(sizeof(CtrlTransportTokenView) == sizeof(uint64_t), "");

}  // namespace

TransportToken::TransportToken(TransportTokenType type) {
  static_assert(sizeof(TransportToken) == sizeof(int64_t), "");
  *reinterpret_cast<int64_t*>(this) = 0;
  type_ = type;
}

/*static*/ TransportToken TransportToken::NewDataTransportToken() {
  static auto* seq_id = new std::atomic<int64_t>();
  TransportToken transport_token(kDataTransportTokenType);
  CHECK_JUST(DataTransportTokenView::MutCast(&transport_token))->set_data_seq_id(++*seq_id);
  return transport_token;
}

/*static*/ Maybe<TransportToken> TransportToken::NewMetaTransportToken() {
  int32_t thread_consistent_unique_id = JUST(GetThisThreadConsistentUniqueId());
  int32_t rank_group_level = JUST(GetCurrentRankGroupLevel());
  static const int kLimit = 128;
  CHECK_GE_OR_RETURN(rank_group_level, 0);
  CHECK_LT_OR_RETURN(rank_group_level, kLimit);
  static thread_local std::array<std::unique_ptr<TransportToken>, kLimit> transport_token_stack;
  auto* current_transport_token = &transport_token_stack[rank_group_level];
  if (!*current_transport_token) {
    const auto& init = JUST(NewMetaTransportToken(thread_consistent_unique_id, rank_group_level));
    current_transport_token->reset(new TransportToken(init));
  }
  return ++**current_transport_token;
}

namespace {

Maybe<bool*> ThreadLocalMutLock4CtrlTransportToken(int32_t thread_consistent_unique_id,
                                                   int32_t rank_group_level, RankGroupCtrlCmd cmd) {
  CHECK_EQ_OR_RETURN(thread_consistent_unique_id, JUST(GetThisThreadConsistentUniqueId()));
  static const int kTransportTokenRankGroupLevelLimit = (1 << kTransportTokenRankGroupLevelBit);
  CHECK_LT_OR_RETURN(rank_group_level, kTransportTokenRankGroupLevelLimit);
  static thread_local std::array<std::array<bool, kSizeOfRankGroupCtrlCmd>,
                                 kTransportTokenRankGroupLevelLimit>
      transport_token_lock;
  return &transport_token_lock[rank_group_level][cmd];
}

}  // namespace

/*static*/ Maybe<TransportToken> TransportToken::AcquireCtrlTransportToken(RankGroupCtrlCmd cmd) {
  int32_t thread_consistent_unique_id = JUST(GetThisThreadConsistentUniqueId());
  int32_t rank_group_level = JUST(GetCurrentRankGroupLevel());
  auto* lock = JUST(
      ThreadLocalMutLock4CtrlTransportToken(thread_consistent_unique_id, rank_group_level, cmd));
  CHECK_OR_RETURN(!*lock);
  static const int kTransportTokenRankGroupLevelLimit = (1 << kTransportTokenRankGroupLevelBit);
  static thread_local std::array<
      std::array<std::unique_ptr<TransportToken>, kSizeOfRankGroupCtrlCmd>,
      kTransportTokenRankGroupLevelLimit>
      transport_token_stack;
  CHECK_GE_OR_RETURN(rank_group_level, 0);
  CHECK_LT_OR_RETURN(rank_group_level, kTransportTokenRankGroupLevelLimit);
  CHECK_GE_OR_RETURN(static_cast<int>(cmd), 0);
  CHECK_LT_OR_RETURN(static_cast<int>(cmd), kSizeOfRankGroupCtrlCmd);
  auto* current_transport_token = &transport_token_stack[rank_group_level][cmd];
  if (!*current_transport_token) {
    const auto& init =
        JUST(NewCtrlTransportToken(cmd, thread_consistent_unique_id, rank_group_level));
    current_transport_token->reset(new TransportToken(init));
  }
  *lock = true;
  return **current_transport_token;
}

Maybe<void> TransportToken::TryAcquireCtrlTransportTokenLock() const {
  if (type() == kCtrlTransportTokenType) {
    auto* lock = JUST(ThreadLocalMutLock4CtrlTransportToken(JUST(thread_consistent_unique_id()),
                                                            JUST(rank_group_level()), JUST(cmd())));
    CHECK_OR_RETURN(!*lock);
    *lock = true;
  }
  return Maybe<void>::Ok();
}

Maybe<void> TransportToken::TryReleaseCtrlTransportTokenLock() const {
  if (type() == kCtrlTransportTokenType) {
    const auto& thread_consistent_unique_id = JUST(this->thread_consistent_unique_id());
    const auto& rank_group_level = JUST(this->rank_group_level());
    const auto& cmd = JUST(this->cmd());
    auto* lock = JUST(
        ThreadLocalMutLock4CtrlTransportToken(thread_consistent_unique_id, rank_group_level, cmd));
    CHECK_OR_RETURN(*lock);
    *lock = false;
  }
  return Maybe<void>::Ok();
}

Maybe<int64_t> TransportToken::thread_consistent_unique_id() const {
  if (type() == kMetaTransportTokenType) {
    return JUST(MetaTransportTokenView::Cast(this))->thread_consistent_unique_id();
  } else if (type() == kCtrlTransportTokenType) {
    return JUST(CtrlTransportTokenView::Cast(this))->thread_consistent_unique_id();
  } else {
    UNIMPLEMENTED_THEN_RETURN();
  }
  UNIMPLEMENTED_THEN_RETURN();
}

Maybe<int64_t> TransportToken::rank_group_level() const {
  if (type() == kMetaTransportTokenType) {
    return JUST(MetaTransportTokenView::Cast(this))->rank_group_level();
  } else if (type() == kCtrlTransportTokenType) {
    return JUST(CtrlTransportTokenView::Cast(this))->rank_group_level();
  } else {
    UNIMPLEMENTED_THEN_RETURN();
  }
  UNIMPLEMENTED_THEN_RETURN();
}

Maybe<RankGroupCtrlCmd> TransportToken::cmd() const {
  return JUST(CtrlTransportTokenView::Cast(this))->cmd();
}

Maybe<void> TransportToken::set_src_rank(int64_t src_rank) {
  CHECK_GE_OR_RETURN(src_rank, 0);
  CHECK_LT_OR_RETURN(src_rank, GetMaxVal<uint16_t>());
  src_rank_ = src_rank;
  return Maybe<void>::Ok();
}

Maybe<void> TransportToken::set_dst_rank(int64_t dst_rank) {
  CHECK_GE_OR_RETURN(dst_rank, 0);
  CHECK_LT_OR_RETURN(dst_rank, GetMaxVal<uint16_t>());
  dst_rank_ = dst_rank;
  return Maybe<void>::Ok();
}

TransportToken::operator uint64_t() const {
  static_assert(sizeof(TransportToken) == sizeof(uint64_t), "");
  return *reinterpret_cast<const uint64_t*>(this);
}

TransportToken& TransportToken::operator++() {
  TransportTokenType transport_token_type = type();
  if (transport_token_type == kDataTransportTokenType) {
    UNIMPLEMENTED();
  } else if (transport_token_type == kMetaTransportTokenType) {
    ++*CHECK_JUST(MetaTransportTokenView::MutCast(this));
  } else if (transport_token_type == kCtrlTransportTokenType) {
    UNIMPLEMENTED();
  } else {
    UNIMPLEMENTED();
  }
  return *this;
}

/*static*/ Maybe<TransportToken> TransportToken::NewMetaTransportToken(
    int32_t thread_consistent_unique_id, int32_t rank_group_level) {
  TransportToken transport_token(kMetaTransportTokenType);
  auto* view = JUST(MetaTransportTokenView::MutCast(&transport_token));
  JUST(view->set_thread_consistent_unique_id(thread_consistent_unique_id));
  JUST(view->set_rank_group_level(rank_group_level));
  return transport_token;
}

/*static*/ Maybe<TransportToken> TransportToken::NewCtrlTransportToken(
    RankGroupCtrlCmd cmd, int32_t thread_consistent_unique_id, int32_t rank_group_level) {
  TransportToken transport_token(kCtrlTransportTokenType);
  auto* view = JUST(CtrlTransportTokenView::MutCast(&transport_token));
  JUST(view->set_thread_consistent_unique_id(thread_consistent_unique_id));
  JUST(view->set_rank_group_level(rank_group_level));
  view->set_cmd(cmd);
  view->set_ctrl_seq_id(0);
  return transport_token;
}

}  // namespace oneflow
