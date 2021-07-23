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
#ifndef ONEFLOW_CORE_TRANSPORT_TRANSPORT_MESSAGE_H_
#define ONEFLOW_CORE_TRANSPORT_TRANSPORT_MESSAGE_H_

#include "oneflow/core/common/platform.h"
#include "oneflow/core/common/util.h"

#ifdef __linux__

namespace oneflow {

enum class TransportMsgType {
  kInvalid = 0,
  kSend = 1,  // send msg from local to remote transport
  kAck = 2,   // this token transmission task is down
};

struct TransportMsg {
  uint64_t token;
  void* src_mem_token;
  void* dst_mem_token;
  std::size_t size;
  int64_t src_machine_id;
  int64_t dst_machine_id;
  TransportMsgType type;
};

}  // namespace oneflow

#endif  // __linux__

#endif  // ONEFLOW_CORE_TRANSPORT_TRANSPORT_MESSAGE_H_
