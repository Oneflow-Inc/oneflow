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
#include "oneflow/core/platform/include/ibv.h"

#include <infiniband/verbs.h>

#include <deque>
#include <memory>
#include <mutex>

#if defined(WITH_RDMA) && defined(OF_PLATFORM_POSIX)

namespace oneflow {

class ActorMsgMR final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ActorMsgMR);
  ActorMsgMR() = delete;
  ActorMsgMR(ibv_mr* mr, char* addr, size_t size) : size_(size) {
    msg_ = reinterpret_cast<ActorMsg*>(addr);
    mr_ = mr;
  }
  ~ActorMsgMR() = default;

  char* addr() { return reinterpret_cast<char*>(msg_); }
  uint32_t lkey() { return mr_->lkey; }
  ActorMsg msg() { return *msg_; }
  void set_msg(const ActorMsg& val) { *msg_ = val; }
  size_t size() { return size_; }

 private:
  size_t size_;
  ibv_mr* mr_;
  ActorMsg* msg_;
};

class IBVerbsQP;

struct WorkRequestId {
  IBVerbsQP* qp;
  int32_t outstanding_sge_cnt;
  void* read_id;
  ActorMsgMR* msg_mr;
};

class MessagePool final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MessagePool);
  MessagePool() = delete;
  ~MessagePool() = default;

  MessagePool(ibv_pd* pd, uint32_t number_of_message)
      : pd_(pd), num_of_message_(number_of_message) {
    RegisterMessagePool();
  }

  void RegisterMessagePool() {
    ActorMsg msg;
    size_t ActorMsgSize = sizeof(msg);
    size_t RegisterMemorySize = ActorMsgSize * (num_of_message_);
    char* addr = (char*)malloc(RegisterMemorySize);
    ibv_mr* mr = ibv::wrapper.ibv_reg_mr_wrap(
        pd_, addr, RegisterMemorySize,
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
    CHECK(mr);
    mr_buf_.push_front(mr);
    for (size_t i = 0; i < num_of_message_; i++) {
      char* split_addr = addr + ActorMsgSize * i;
      ActorMsgMR* msg_mr = new ActorMsgMR(mr, split_addr, ActorMsgSize);
      message_buf_.push_front(msg_mr);
    }
  }

  ActorMsgMR* GetMessage() {
    if (IsEmpty() == false) {
      return GetMessageFromBuf();
    } else {
      RegisterMessagePool();
      return GetMessageFromBuf();
    }
  }

  ActorMsgMR* GetMessageFromBuf() {
    std::unique_lock<std::mutex> msg_buf_lck(message_buf_mutex_);
    ActorMsgMR* msg_mr = message_buf_.front();
    message_buf_.pop_front();
    return msg_mr;
  }

  void PutMessage(ActorMsgMR* msg_mr) {
    std::unique_lock<std::mutex> msg_buf_lck(message_buf_mutex_);
    message_buf_.push_front(msg_mr);
  }

  bool IsEmpty() {
    std::unique_lock<std::mutex> msg_buf_lck(message_buf_mutex_);
    return message_buf_.empty() == true;
  }

  void FreeMr() {
    while (mr_buf_.empty() == false) {
      ibv_mr* mr = mr_buf_.front();
      mr_buf_.pop_front();
      CHECK_EQ(ibv::wrapper.ibv_dereg_mr(mr), 0);
    }
  }

 private:
  ibv_pd* pd_;
  size_t num_of_message_;
  std::mutex message_buf_mutex_;
  std::deque<ActorMsgMR*> message_buf_;
  std::deque<ibv_mr*> mr_buf_;
};

struct IBVerbsCommNetRMADesc;

class IBVerbsQP final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IBVerbsQP);
  IBVerbsQP() = delete;
  IBVerbsQP(ibv_context*, ibv_pd*, uint8_t port_num, ibv_cq* send_cq, ibv_cq* recv_cq,
            MessagePool* msg_buf);
  ~IBVerbsQP();

  uint32_t qp_num() const { return qp_->qp_num; }
  void Connect(const IBVerbsConnectionInfo& peer_info);
  void PostAllRecvRequest();
  void GetActorMsgMRFromMessagePool();
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
  MessagePool* msg_Pool_buf_;
};

}  // namespace oneflow

#endif  // WITH_RDMA && OF_PLATFORM_POSIX

#endif  // ONEFLOW_CORE_COMM_NETWORK_IBVERBS_IBVERBS_QP_H_
