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

#include <cstdint>
#include "oneflow/core/comm_network/ibverbs/ibverbs_memory_desc.h"
#include "oneflow/core/actor/actor_message.h"

#if defined(WITH_RDMA) && defined(OF_PLATFORM_POSIX)

namespace oneflow {

class ActorMsgMR final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ActorMsgMR);
  ActorMsgMR() = delete;
  ActorMsgMR(ibv_pd* pd) { mem_desc_.reset(new IBVerbsMemDesc(pd, &msg_, sizeof(msg_))); }
  ActorMsgMR(IBVerbsMemDesc * mem_desc){
    mem_desc_.reset(mem_desc);
  }
  ~ActorMsgMR() { mem_desc_.reset(); }

  const ActorMsg& msg() const { return msg_; }
  void set_msg(const ActorMsg& val) { msg_ = val; }
  const IBVerbsMemDesc& mem_desc() const { return *mem_desc_; }

 private:
  ActorMsg msg_;
  std::unique_ptr<IBVerbsMemDesc> mem_desc_;
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
    MessagePool() = default; //todo:这里可能要修改
    ~MessagePool() = default;//todo:这里可能要修改
    MessagePool(ibv_pd* pd,uint32_t size, uint32_t number_of_message): pd_(pd),num_of_message_(number_of_message) {}
    //以后这里可以切割内存，注册一块大的，再不断的分割
    void RegisterMessagePool() {
      for(int i =0; i < num_of_message_; i++){
    //  void * addr = malloc(size_);
    //  IBVerbsMemDesc * mem_desc =new IBVerbsMemDesc(pd_,addr,size_);
      ActorMsgMR * msg_mr = new ActorMsgMR(pd_);
      message_buf_.push(msg_mr);
    }
  }

  ActorMsgMR *  GetMessage(){
    if(isEmpty() == false)  {
      return GetMessageFromBuf();
    } else {
      RegisterMessagePool();
      return GetMessageFromBuf();
    }
  }

  ActorMsgMR * GetMessageFromBuf() {
      ActorMsgMR * msg_mr = std::move(message_buf_.front());
      message_buf_.pop();
      return msg_mr;
  }
    /*ActorMsgMR * GetMessage(){
      if(message_buf_.empty() == false) {
          ActorMsgMR * msg_mr  =message_buf_.front();
          message_buf_.pop();
          return msg_mr;
      } else {
        //register a big memory 
        addr_ = malloc( size_ * num_of_message_);//申请内存空间
        mem_desc_ = new IBVerbsMemDesc(pd_,addr_, size_ * num_of_message_);//给这一块内存空间注册内
        //切割内存
        const ibv_mr* mr = mem_desc_->mr();
         for(int i = 0; i < num_of_message_; i++){
           ibv_mr * mr1 =(ibv_mr*) (mr + size_ * i);
           void* addr =(void*) ((char*)addr_ + size_* i);
           IBVerbsMemDesc * mem_desc =new  IBVerbsMemDesc(mr1,addr,size_);
            ActorMsgMR * actorMr = new ActorMsgMR(mem_desc);
            message_buf_.push(actorMr);
         }
          ActorMsgMR * msg_mr  =message_buf_.front();
          message_buf_.pop();
          return msg_mr;
      }
      
    }
    void PutMessage(const ActorMsg & msg){
     // message_buf_.push()
      if(message_buf_.empty() == false) {
        ActorMsgMR * msg_mr  =message_buf_.front();
        msg_mr->set_msg(msg);
      } else {
        //register a big memory 
        addr_ = malloc( size_ * num_of_message_);//申请内存空间
        mem_desc_ = new IBVerbsMemDesc(pd_,addr_, size_ * num_of_message_);//给这一块内存空间注册内存
        //切割内存
        const ibv_mr* mr = mem_desc_->mr();
         for(int i = 0; i < num_of_message_; i++){
           ibv_mr * mr1 =(ibv_mr*) (mr + size_ * i);
           void* addr =(void*) ((char*)addr_ + size_* i);
           IBVerbsMemDesc * mem_desc =new  IBVerbsMemDesc(mr1,addr,size_);
           ActorMsgMR * actorMr = new ActorMsgMR(mem_desc);
           message_buf_.push(actorMr);
         }
        ActorMsgMR * msg_mr  =message_buf_.front();
        msg_mr->set_msg(msg);
      }
    }*/
    void PutMessage(ActorMsgMR * msg_mr) {
      message_buf_.push(msg_mr);
    }
    std::queue<ActorMsgMR*> getMessageBuf() {
      return message_buf_;
    }

    bool isEmpty() {
      return message_buf_.empty();
    }

  private:
    ibv_pd* pd_;
 //   uint32_t size_;
    uint32_t num_of_message_;
    std::queue<ActorMsgMR*> message_buf_;
};
struct IBVerbsCommNetRMADesc;

class IBVerbsQP final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IBVerbsQP);
  IBVerbsQP() = delete;
  IBVerbsQP(ibv_context*, ibv_pd*, uint8_t port_num, ibv_cq* send_cq, ibv_cq* recv_cq);
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
 // std::vector<ActorMsgMR*> recv_msg_buf_;

  std::mutex send_msg_buf_mtx_;
  std::queue<ActorMsgMR*> send_msg_buf_;
  std::mutex pending_send_wr_mutex_;
  uint32_t num_outstanding_send_wr_;
  uint32_t max_outstanding_send_wr_;
  std::queue<std::pair<ibv_send_wr, ibv_sge>> pending_send_wr_queue_;
  
  MessagePool  *  recv_msg_buf_;
  MessagePool  * sendMsgBuf_;
};

}  // namespace oneflow

#endif  // WITH_RDMA && OF_PLATFORM_POSIX

#endif  // ONEFLOW_CORE_COMM_NETWORK_IBVERBS_IBVERBS_QP_H_
