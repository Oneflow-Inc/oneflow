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

namespace oneflow {

enum RpcTokenType {
  // Begin
  kDataRpcTokenType = 0, // e.g. for tensor data transportation
  kOpTensorMetaRpcTokenType, // e.g. for tensor shape synchronizing or checking
  kSystemRpcTokenType, // e.g. for rank_group or thread checking
  kExtendedRpcTokenType, // for compatibility
  // End
  kRpcTokenTypeSize,
};

static_assert(kRpcTokenTypeSize <= 4, "");

class RpcToken final {
 public:
  RpcToken(RpcTokenType type, uint32_t minor) : major_(major), minor_(minor) {}
  RpcToken(const RpcToken&) = default;
  RpcToken(RpcToken&) = default;
  ~RpcToken() = default;

  int64_t src_machine_id() const { return src_machine_id_; }
  int64_t dst_machine_id() const { return dst_machine_id_; }
  RpcTokenType type() const { return static_cast<RpcTokenType>(type_); }
  int64_t consistent_thread_id() const { return consistent_thread_id_; }
  int64_t rank_group_id() const { return rank_group_id_; }
  int64_t seq_id() const { return seq_id_; }

  operator uint64_t() const;
  RpcToken& operator++();

 private:

  uint16_t src_machine_id_;
  uint16_t dst_machine_id_;
  uint32_t type_:2;
  uint32_t consistent_thread_id_:3;
  uint32_t rank_group_id_:3;
  uint32_t seq_id_:24;
};

template<>
struct IsScalarType<RpcToken> final {
  static const bool value = true;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_RPC_TOKEN_H_
