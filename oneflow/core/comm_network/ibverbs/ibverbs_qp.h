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
#ifndef ONEFLOW_CORE_COMM_NETWORK_IBVERBS_IBVERBS_QP_H_
#define ONEFLOW_CORE_COMM_NETWORK_IBVERBS_IBVERBS_QP_H_

#include "oneflow/core/comm_network/ibverbs/ibverbs_memory_desc.h"
#include "oneflow/core/actor/actor_message.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/platform/include/ibv.h"

#if defined(WITH_RDMA) && defined(OF_PLATFORM_POSIX)

namespace oneflow {

class ActorMsgMR final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ActorMsgMR);
  ActorMsgMR() = delete;
  ActorMsgMR(ibv_mr* mr, char* addr, size_t actor_msg_size) {
    CHECK(actor_msg_size >= sizeof(ActorMsg));
    actor_msg_size_ = actor_msg_size;
    message_ = reinterpret_cast<ActorMsg*>(addr);
    mr_ = mr;
  }
  ~ActorMsgMR() = default;

  void* addr() { return reinterpret_cast<void*>(message_); }
  uint32_t lkey() const { return mr_->lkey; }
  ActorMsg msg() const { return *message_; }
  void set_msg(const ActorMsg& val) { *message_ = val; }
  size_t actor_msg_size() const { return actor_msg_size_; }

 private:
  size_t actor_msg_size_;
  ibv_mr* mr_;
  ActorMsg* message_;
};

class IBVerbsQP;

struct WorkRequestId {
  IBVerbsQP* qp;
  int32_t outstanding_sge_cnt;
  void* read_id;
  ActorMsgMR* msg_mr;
};

class IBVerbsMessagePool;

struct IBVerbsCommNetRMADesc;

class IBVerbsQP final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IBVerbsQP);
  IBVerbsQP() = delete;
  IBVerbsQP(ibv_context*, ibv_pd*, uint8_t port_num, ibv_cq* send_cq, ibv_cq* recv_cq,
            IBVerbsMessagePool* message_pool);
  ~IBVerbsQP();

  uint32_t qp_num() const { return qp_->qp_num; }
  void Connect(const IBVerbsConnectionInfo& peer_info);
  void PostAllRecvRequest();

  void PostReadRequest(const IBVerbsCommNetRMADesc& remote_mem, const IBVerbsMemDesc& local_mem,
                       void* read_id);
  void PostSendRequest(const ActorMsg& msg);

  void ReadDone(WorkRequestId*);
  void SendDone(WorkRequestId*);
  void RecvDone(WorkRequestId*);

 private:
  void EnqueuePostSendReadWR(ibv_send_wr wr, ibv_sge sge);
  void PostPendingSendWR();
  WorkRequestId* NewWorkRequestId();
  void DeleteWorkRequestId(WorkRequestId* wr_id);
  ActorMsgMR* GetOneSendMsgMRFromBuf();
  void PostRecvRequest(ActorMsgMR*);

  ibv_context* ctx_;
  ibv_pd* pd_;
  uint8_t port_num_;
  ibv_qp* qp_;

  std::mutex pending_send_wr_mutex_;
  uint32_t num_outstanding_send_wr_;
  uint32_t max_outstanding_send_wr_;
  std::queue<std::pair<ibv_send_wr, ibv_sge>> pending_send_wr_queue_;
  IBVerbsMessagePool* message_pool_;
  uint32_t queue_depth_;
};

}  // namespace oneflow

#endif  // WITH_RDMA && OF_PLATFORM_POSIX

#endif  // ONEFLOW_CORE_COMM_NETWORK_IBVERBS_IBVERBS_QP_H_
