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
#include "oneflow/core/framework/rpc_token.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/thread/consistent_unique_id.h"
#include "oneflow/core/framework/rank_group_rpc_util.h"

namespace oneflow {

namespace {

class DataRpcTokenView final {
 public:
  static Maybe<DataRpcTokenView*> MutCast(RpcToken* rpc_token) {
    CHECK_EQ_OR_RETURN(rpc_token->type(), kDataRpcTokenType);
    return reinterpret_cast<DataRpcTokenView*>(rpc_token);
  }

  void set_data_seq_id(int64_t seq_id) { data_seq_id_ = seq_id; }

 private:
  uint16_t src_rank_;
  uint16_t dst_rank_;
  uint32_t type_ : 2;  // RpcTokenType
  uint32_t data_seq_id_ : 30;
};
static_assert(sizeof(DataRpcTokenView) == sizeof(uint64_t), "");

class MetaRpcTokenView final {
 public:
  int64_t thread_consistent_unique_id() const { return thread_consistent_unique_id_; }
  int64_t rank_group_level() const { return rank_group_level_; }

  static Maybe<MetaRpcTokenView*> MutCast(RpcToken* rpc_token) {
    CHECK_EQ_OR_RETURN(rpc_token->type(), kMetaRpcTokenType);
    return reinterpret_cast<MetaRpcTokenView*>(rpc_token);
  }

  static Maybe<const MetaRpcTokenView*> Cast(const RpcToken* rpc_token) {
    CHECK_EQ_OR_RETURN(rpc_token->type(), kMetaRpcTokenType);
    return reinterpret_cast<const MetaRpcTokenView*>(rpc_token);
  }

  Maybe<void> set_thread_consistent_unique_id(int8_t val) {
    CHECK_GE_OR_RETURN(val, 0);
    CHECK_LT_OR_RETURN(val, 1 << kRpcTokenThreadConsistentUIdBit);
    thread_consistent_unique_id_ = val;
    return Maybe<void>::Ok();
  }

  Maybe<void> set_rank_group_level(int32_t val) {
    CHECK_GE_OR_RETURN(val, 0);
    CHECK_LT_OR_RETURN(val, 1 << kRpcTokenRankGroupLevelBit);
    rank_group_level_ = val;
    return Maybe<void>::Ok();
  }

  MetaRpcTokenView& operator++() {
    ++low_meta_seq_id_;
    if (low_meta_seq_id_ == 0) { ++high_meta_seq_id_; }
    return *this;
  }

 private:
  uint16_t src_rank_;
  uint16_t dst_rank_;
  uint8_t type_ : 2;  // RpcTokenType
  uint8_t thread_consistent_unique_id_ : kRpcTokenThreadConsistentUIdBit;
  uint8_t rank_group_level_ : kRpcTokenRankGroupLevelBit;
  uint8_t high_meta_seq_id_;
  uint16_t low_meta_seq_id_;
};
static_assert(sizeof(MetaRpcTokenView) == sizeof(uint64_t), "");

class CtrlRpcTokenView final {
 public:
  int64_t thread_consistent_unique_id() const { return thread_consistent_unique_id_; }
  int64_t rank_group_level() const { return rank_group_level_; }

  static Maybe<CtrlRpcTokenView*> MutCast(RpcToken* rpc_token) {
    CHECK_EQ_OR_RETURN(rpc_token->type(), kCtrlRpcTokenType);
    return reinterpret_cast<CtrlRpcTokenView*>(rpc_token);
  }

  static Maybe<const CtrlRpcTokenView*> Cast(const RpcToken* rpc_token) {
    CHECK_EQ_OR_RETURN(rpc_token->type(), kCtrlRpcTokenType);
    return reinterpret_cast<const CtrlRpcTokenView*>(rpc_token);
  }

  Maybe<void> set_thread_consistent_unique_id(int8_t val) {
    CHECK_GE_OR_RETURN(val, 0);
    CHECK_LT_OR_RETURN(val, 1 << kRpcTokenThreadConsistentUIdBit);
    thread_consistent_unique_id_ = val;
    return Maybe<void>::Ok();
  }
  Maybe<void> set_rank_group_level(int32_t val) {
    CHECK_GE_OR_RETURN(val, 0);
    CHECK_LT_OR_RETURN(val, 1 << kRpcTokenRankGroupLevelBit);
    rank_group_level_ = val;
    return Maybe<void>::Ok();
  }

  RankGroupRpcCmd cmd() const { return static_cast<RankGroupRpcCmd>(cmd_); }

  void set_cmd(RankGroupRpcCmd cmd) {
    static_assert(kSizeOfRankGroupRpcCmd < (1 << 8), "");
    cmd_ = static_cast<int8_t>(cmd);
  }

  void set_ctrl_seq_id(int32_t val) { ctrl_seq_id_ = val; }

 private:
  uint16_t src_rank_;
  uint16_t dst_rank_;
  uint8_t type_ : 2;  // RpcTokenType
  uint8_t thread_consistent_unique_id_ : kRpcTokenThreadConsistentUIdBit;
  uint8_t rank_group_level_ : kRpcTokenRankGroupLevelBit;
  uint8_t cmd_;
  uint16_t ctrl_seq_id_;
};
static_assert(sizeof(CtrlRpcTokenView) == sizeof(uint64_t), "");

}  // namespace

RpcToken::RpcToken(RpcTokenType type) {
  static_assert(sizeof(RpcToken) == sizeof(int64_t), "");
  *reinterpret_cast<int64_t*>(this) = 0;
  type_ = type;
}

/*static*/ RpcToken RpcToken::NewDataRpcToken() {
  static auto* seq_id = new std::atomic<int64_t>();
  RpcToken rpc_token(kDataRpcTokenType);
  CHECK_JUST(DataRpcTokenView::MutCast(&rpc_token))->set_data_seq_id(++*seq_id);
  return rpc_token;
}

/*static*/ Maybe<RpcToken> RpcToken::NewMetaRpcToken() {
  int32_t thread_consistent_unique_id = JUST(GetThisThreadConsistentUniqueId());
  int32_t rank_group_level = JUST(GetCurrentRankGroupLevel());
  static const int kLimit = 128;
  CHECK_GE_OR_RETURN(rank_group_level, 0);
  CHECK_LT_OR_RETURN(rank_group_level, kLimit);
  static thread_local std::array<std::unique_ptr<RpcToken>, kLimit> rpc_token_stack;
  auto* current_rpc_token = &rpc_token_stack[rank_group_level];
  if (!*current_rpc_token) {
    const auto& init = JUST(NewMetaRpcToken(thread_consistent_unique_id, rank_group_level));
    current_rpc_token->reset(new RpcToken(init));
  }
  return ++**current_rpc_token;
}

namespace {

Maybe<bool*> ThreadLocalMutLock4CtrlRpcToken(int32_t thread_consistent_unique_id,
                                             int32_t rank_group_level, RankGroupRpcCmd cmd) {
  CHECK_EQ_OR_RETURN(thread_consistent_unique_id, JUST(GetThisThreadConsistentUniqueId()));
  static const int kRpcTokenRankGroupLevelLimit = (1 << kRpcTokenRankGroupLevelBit);
  CHECK_LT_OR_RETURN(rank_group_level, kRpcTokenRankGroupLevelLimit);
  static thread_local std::array<std::array<bool, kSizeOfRankGroupRpcCmd>,
                                 kRpcTokenRankGroupLevelLimit>
      rpc_token_lock;
  return &rpc_token_lock[rank_group_level][cmd];
}

}  // namespace

/*static*/ Maybe<RpcToken> RpcToken::AcquireCtrlRpcToken(RankGroupRpcCmd cmd) {
  int32_t thread_consistent_unique_id = JUST(GetThisThreadConsistentUniqueId());
  int32_t rank_group_level = JUST(GetCurrentRankGroupLevel());
  auto* lock =
      JUST(ThreadLocalMutLock4CtrlRpcToken(thread_consistent_unique_id, rank_group_level, cmd));
  CHECK_OR_RETURN(!*lock);
  static const int kRpcTokenRankGroupLevelLimit = (1 << kRpcTokenRankGroupLevelBit);
  static thread_local std::array<std::array<std::unique_ptr<RpcToken>, kSizeOfRankGroupRpcCmd>,
                                 kRpcTokenRankGroupLevelLimit>
      rpc_token_stack;
  CHECK_GE_OR_RETURN(rank_group_level, 0);
  CHECK_LT_OR_RETURN(rank_group_level, kRpcTokenRankGroupLevelLimit);
  CHECK_GE_OR_RETURN(static_cast<int>(cmd), 0);
  CHECK_LT_OR_RETURN(static_cast<int>(cmd), kSizeOfRankGroupRpcCmd);
  auto* current_rpc_token = &rpc_token_stack[rank_group_level][cmd];
  if (!*current_rpc_token) {
    const auto& init = JUST(NewCtrlRpcToken(cmd, thread_consistent_unique_id, rank_group_level));
    current_rpc_token->reset(new RpcToken(init));
  }
  *lock = true;
  return **current_rpc_token;
}

Maybe<void> RpcToken::ReleaseCtrlRpcToken() const {
  auto* lock = JUST(ThreadLocalMutLock4CtrlRpcToken(JUST(thread_consistent_unique_id()),
                                                    JUST(rank_group_level()), JUST(cmd())));
  CHECK_OR_RETURN(*lock);
  *lock = false;
  return Maybe<void>::Ok();
}

Maybe<int64_t> RpcToken::thread_consistent_unique_id() const {
  if (type() == kMetaRpcTokenType) {
    return JUST(MetaRpcTokenView::Cast(this))->thread_consistent_unique_id();
  } else if (type() == kCtrlRpcTokenType) {
    return JUST(CtrlRpcTokenView::Cast(this))->thread_consistent_unique_id();
  } else {
    UNIMPLEMENTED_THEN_RETURN();
  }
  UNIMPLEMENTED_THEN_RETURN();
}

Maybe<int64_t> RpcToken::rank_group_level() const {
  if (type() == kMetaRpcTokenType) {
    return JUST(MetaRpcTokenView::Cast(this))->rank_group_level();
  } else if (type() == kCtrlRpcTokenType) {
    return JUST(CtrlRpcTokenView::Cast(this))->rank_group_level();
  } else {
    UNIMPLEMENTED_THEN_RETURN();
  }
  UNIMPLEMENTED_THEN_RETURN();
}

Maybe<RankGroupRpcCmd> RpcToken::cmd() const { return JUST(CtrlRpcTokenView::Cast(this))->cmd(); }

Maybe<void> RpcToken::set_src_rank(int64_t src_rank) {
  CHECK_GE_OR_RETURN(src_rank, 0);
  CHECK_LT_OR_RETURN(src_rank, GetMaxVal<uint16_t>());
  src_rank_ = src_rank;
  return Maybe<void>::Ok();
}

Maybe<void> RpcToken::set_dst_rank(int64_t dst_rank) {
  CHECK_GE_OR_RETURN(dst_rank, 0);
  CHECK_LT_OR_RETURN(dst_rank, GetMaxVal<uint16_t>());
  dst_rank_ = dst_rank;
  return Maybe<void>::Ok();
}

RpcToken::operator uint64_t() const {
  static_assert(sizeof(RpcToken) == sizeof(uint64_t), "");
  return *reinterpret_cast<const uint64_t*>(this);
}

RpcToken& RpcToken::operator++() {
  RpcTokenType rpc_token_type = type();
  if (rpc_token_type == kDataRpcTokenType) {
    UNIMPLEMENTED();
  } else if (rpc_token_type == kMetaRpcTokenType) {
    ++*CHECK_JUST(MetaRpcTokenView::MutCast(this));
  } else if (rpc_token_type == kCtrlRpcTokenType) {
    UNIMPLEMENTED();
  } else {
    UNIMPLEMENTED();
  }
  return *this;
}

/*static*/ Maybe<RpcToken> RpcToken::NewMetaRpcToken(int32_t thread_consistent_unique_id,
                                                     int32_t rank_group_level) {
  RpcToken rpc_token(kMetaRpcTokenType);
  auto* view = JUST(MetaRpcTokenView::MutCast(&rpc_token));
  JUST(view->set_thread_consistent_unique_id(thread_consistent_unique_id));
  JUST(view->set_rank_group_level(rank_group_level));
  return rpc_token;
}

/*static*/ Maybe<RpcToken> RpcToken::NewCtrlRpcToken(RankGroupRpcCmd cmd,
                                                     int32_t thread_consistent_unique_id,
                                                     int32_t rank_group_level) {
  RpcToken rpc_token(kCtrlRpcTokenType);
  auto* view = JUST(CtrlRpcTokenView::MutCast(&rpc_token));
  JUST(view->set_thread_consistent_unique_id(thread_consistent_unique_id));
  JUST(view->set_rank_group_level(rank_group_level));
  view->set_cmd(cmd);
  view->set_ctrl_seq_id(0);
  return rpc_token;
}

}  // namespace oneflow
