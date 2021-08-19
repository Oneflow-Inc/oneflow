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

#include <functional>
#include "oneflow/core/common/type_traits.h"
#include "oneflow/core/common/maybe.h"

namespace oneflow {

const static int kTransportTokenTypeBit = 8;
const static int kTransportTokenThreadConsistentUIdBit = 8;
const static int kTransportTokenRankGroupLevelBit = 16;

enum TransportTokenType {
  // Begin
  kTransportTokenTypeInvalid = 0,
  kTransportTokenTypeData,  // e.g. for tensor data transportation
  kTransportTokenTypeMeta,  // e.g. for tensor consistent id generating
  kTransportTokenTypeSyncSymbolParallelDesc,
  kTransportTokenTypeSyncSymbolNdSbp,
  kTransportTokenTypeSyncSymbolConsistentTensorMeta,
  kTransportTokenTypeCheckRankGroupConsistency,
  kTransportTokenTypeCheckTensorConsistency,
  kTransportTokenTypeSyncLocalShapeDtype,
  // End
  kTransportTokenTypeSize,
};

static_assert(kTransportTokenTypeSize <= (1 << kTransportTokenTypeBit), "");

class TransportToken;

template<>
struct IsScalarType<TransportToken> final {
  static const bool value = true;
};

class TransportToken final {
 public:
  TransportToken() : TransportToken(kTransportTokenTypeInvalid, 0, 0) {}
  TransportToken(const TransportToken&) = default;
  TransportToken(TransportToken&) = default;
  ~TransportToken() = default;

  static Maybe<TransportToken> NewTransportToken(TransportTokenType type);

  static constexpr size_t MaxNumberOfThreadConsistentUId() {
    return (1 << kTransportTokenThreadConsistentUIdBit);
  }

  Maybe<void> CheckThreadConsistentId() const;
  Maybe<void> CheckRankGroupLevel() const;
  bool operator==(const TransportToken& other) const {
    return static_cast<uint64_t>(*this) == static_cast<uint64_t>(other);
  }

  // Getters
  TransportTokenType type() const { return static_cast<TransportTokenType>(type_); }
  uint8_t thread_consistent_id() const { return thread_consistent_id_; }
  uint16_t rank_group_level() const { return rank_group_level_; }
  uint32_t seq_id() const { return seq_id_; }

  operator uint64_t() const { return *reinterpret_cast<const uint64_t*>(this); }

  TransportToken& operator++() {
    ++seq_id_;
    return *this;
  }

 private:
  TransportToken(TransportTokenType type, uint8_t thread_consistent_id, uint16_t rank_group_level)
      : type_(static_cast<uint8_t>(type)),
        thread_consistent_id_(thread_consistent_id),
        rank_group_level_(rank_group_level) {}

  uint8_t type_;  // TransportTokenType
  uint8_t thread_consistent_id_;
  uint16_t rank_group_level_;
  uint32_t seq_id_;
};
static_assert(sizeof(TransportToken) == sizeof(uint64_t), "");

}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::TransportToken> {
  size_t operator()(const oneflow::TransportToken& token) const {
    return std::hash<uint64_t>()(static_cast<uint64_t>(token));
  }
};

}  // namespace std

#endif  // ONEFLOW_CORE_FRAMEWORK_RPC_TOKEN_H_
