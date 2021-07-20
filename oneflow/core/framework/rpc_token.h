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

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/common/type_traits.h"

namespace oneflow {

class ParallelDesc;

enum RpcTokenCmdLocalMajor {
  // Begin
  kCheckingParallelConfSizeCmdLocalMajor = 0,
  kInitializingPlacementCmdLocalMajor,
  kDataSendRecvCmdLocalMajor,
  // End
  kRpcTokenCmdLocalMajorSize,
};

class RpcToken final {
 public:
  RpcToken(uint32_t major, uint32_t minor) : major_(major), minor_(minor) {}
  RpcToken(const RpcToken&) = default;
  RpcToken(RpcToken&) = default;
  ~RpcToken() = default;

  static const uint32_t kStartTokenMajor4Cmd = kRpcTokenCmdLocalMajorSize;
  static const uint32_t kStartTokenMajor4Placement = 4096;

  uint32_t major() const { return major_; }
  uint32_t minor() const { return minor_; }
  operator uint64_t() const { return (static_cast<uint64_t>(major_) << 32) + minor_; }

  RpcToken& operator++() {
    ++minor_;
    return *this;
  }

 private:
  uint32_t major_;
  uint32_t minor_;
};

static_assert(sizeof(RpcToken) == sizeof(uint64_t), "");

template<>
struct IsScalarType<RpcToken> final {
  static const bool value = true;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_RPC_TOKEN_H_
