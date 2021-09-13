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
#ifndef ONEFLOW_CORE_COMM_NETWORK_IBVERBS_IBVERBS_MESSAGE_POOL_H_
#define ONEFLOW_CORE_COMM_NETWORK_IBVERBS_IBVERBS_MESSAGE_POOL_H_

#include "oneflow/core/actor/actor_message.h"
#include "oneflow/core/platform/include/ibv.h"
#include "oneflow/core/comm_network/ibverbs/ibverbs_qp.h"

#if defined(WITH_RDMA) && defined(OF_PLATFORM_POSIX)

namespace oneflow {

class IBVerbsMessagePool final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IBVerbsMessagePool);
  IBVerbsMessagePool() = delete;
  ~IBVerbsMessagePool();
  IBVerbsMessagePool(ibv_pd* pd, uint32_t num_msg_per_bluk_allocation);
  
  ActorMsgMR* GetMessage();
  void PutMessage(ActorMsgMR* msg_mr);

 private:
  void BulkAllocMessage();
  ActorMsgMR* GetMessageFromBuf();
  ibv_pd* pd_;
  size_t num_msg_per_bluk_allocation_;
  std::mutex message_buf_mutex_;
  std::deque<ActorMsgMR*> message_buf_;
  std::deque<ibv_mr*> ibv_mr_buf_;
  std::deque<char*> memory_buf_;
};

}  // namespace oneflow

#endif  // WITH_RDMA && OF_PLATFORM_POSIX

#endif  // ONEFLOW_CORE_COMM_NETWORK_IBVERBS_IBVERBS_MESSAGE_POOL_H_
