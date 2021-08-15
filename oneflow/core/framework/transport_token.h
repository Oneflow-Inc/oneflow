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
#ifndef ONEFLOW_CORE_FRAMEWORK_RPC_TOKEN_H_
#define ONEFLOW_CORE_FRAMEWORK_RPC_TOKEN_H_

#include "oneflow/core/common/type_traits.h"
#include "oneflow/core/common/maybe.h"

namespace oneflow {

const static int kTransportTokenTypeBit = 2;
const static int kTransportTokenThreadConsistentUIdBit = 3;
const static int kTransportTokenRankGroupLevelBit = 3;

enum TransportTokenType {
  // Begin
  kInvalidTransportTokenType = 0,
  kDataTransportTokenType,  // e.g. for tensor data transportation
  kMetaTransportTokenType,  // e.g. for tensor meta checking
  kCtrlTransportTokenType,  // e.g. for rank_group or thread checking. see RankGroupCtrlCmd
  // End
  kTransportTokenTypeSize,
};

static_assert(kTransportTokenTypeSize <= (1 << kTransportTokenTypeBit), "");

enum RankGroupCtrlCmd {
  // Begin
  kRankGroupCtrlCmdInvalid = 0,
  kRankGroupCtrlCmdSyncSymbolParallelDesc,
  kRankGroupCtrlCmdSyncSymbolNdSbp,
  kRankGroupCtrlCmdSyncSymbolConsistentTensorMeta,
  kRankGroupCtrlCmdCheckRankGroupConsistency,
  kRankGroupCtrlCmdCheckTensorConsistency,
  kRankGroupCtrlCmdSyncLocalShapeDtype,
  // End
  kSizeOfRankGroupCtrlCmd
};

class TransportToken;

template<>
struct IsScalarType<TransportToken> final {
  static const bool value = true;
};

class TransportToken final {
 public:
  TransportToken() : TransportToken(kInvalidTransportTokenType) {}
  TransportToken(const TransportToken&) = default;
  TransportToken(TransportToken&) = default;
  ~TransportToken() = default;

  static TransportToken NewDataTransportToken();
  static Maybe<TransportToken> NewMetaTransportToken();
  static Maybe<TransportToken> AcquireCtrlTransportToken(RankGroupCtrlCmd cmd);
  Maybe<void> TryAcquireCtrlTransportTokenLock() const;
  Maybe<void> TryReleaseCtrlTransportTokenLock() const;

  static constexpr size_t MaxNumberOfThreadConsistentUId() {
    return (1 << kTransportTokenThreadConsistentUIdBit);
  }

  // Getters
  int64_t src_rank() const { return src_rank_; }
  int64_t dst_rank() const { return dst_rank_; }
  TransportTokenType type() const { return static_cast<TransportTokenType>(type_); }
  Maybe<int64_t> thread_consistent_unique_id() const;
  Maybe<int64_t> rank_group_level() const;
  Maybe<RankGroupCtrlCmd> cmd() const;

  // Setters
  Maybe<void> set_src_rank(int64_t src_rank);
  Maybe<void> set_dst_rank(int64_t dst_rank);

  operator uint64_t() const;
  TransportToken& operator++();

 private:
  explicit TransportToken(TransportTokenType type);

  static Maybe<TransportToken> NewMetaTransportToken(int32_t thread_consistent_unique_id,
                                                     int32_t rank_group_level);
  static Maybe<TransportToken> NewCtrlTransportToken(RankGroupCtrlCmd cmd,
                                                     int32_t thread_consistent_unique_id,
                                                     int32_t rank_group_level);

  uint16_t src_rank_;
  uint16_t dst_rank_;
  uint32_t type_ : 2;  // TransportTokenType
  uint32_t opaque_ids_ : 30;
};
static_assert(sizeof(TransportToken) == sizeof(uint64_t), "");

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_RPC_TOKEN_H_
