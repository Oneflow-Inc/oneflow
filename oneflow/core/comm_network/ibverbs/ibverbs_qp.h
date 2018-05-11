#ifndef ONEFLOW_CORE_COMM_NETWORK_IBVERBS_IBVERBS_QP_H_
#define ONEFLOW_CORE_COMM_NETWORK_IBVERBS_IBVERBS_QP_H_

#include "oneflow/core/comm_network/ibverbs/ibverbs_memory_desc.h"
#include "oneflow/core/actor/actor_message.h"

#if defined(WITH_RDMA) && defined(PLATFORM_POSIX)

namespace oneflow {

class ActorMsgMR final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ActorMsgMR);
  ActorMsgMR() = delete;
  ActorMsgMR(ibv_pd* pd) { mem_desc_.reset(new IBVerbsMemDesc(pd, &msg_, sizeof(msg_))); }
  ~ActorMsgMR() { mem_desc_.reset(); }

  const ActorMsg& msg() const { return msg_; }
  void set_msg(const ActorMsg& val) { msg_ = val }
  const IBVerbsMemDesc& mem_desc() const { return mem_desc_; }

 private:
  ActorMsg msg_;
  std::unique_ptr<IBVerbsMemDesc> mem_desc_;

};

class IBVerbsQP final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IBVerbsQP);
  IBVerbsQP() = delete;
  IBVerbsQP(ibv_context*, ibv_pd*, ibv_cq* send_cq, ibv_cq* recv_cq);
  ~IBVerbsQP();

  void Connect(const IBVerbsConnectionInfo& peer_info);

  void PostReadRequest(const IBVerbsMemDescProto& remote_mem, const IBVerbsMemDesc& local_mem,
                       void* read_ctx);

  void PostSendRequest(const ActorMsg& msg);
  void SendDone(uint64_t wr_id);
  void RecvDone(uint64_t wr_id);

 private:
  void PostRecvActorMsgRequest(ActorMsgMR*);
  ActorMsgMR* GetOneSendMsgMRFromBuf();

  ibv_context* ctx_;
  ibv_pd* pd_;
  std::queue<ActorMsgMR*> send_msg_buf_;
  std::vector<ActorMsgMR*> recv_msg_buf_;
  ibv_qp* qp_;

};

}  // namespace oneflow

#endif  // WITH_RDMA && PLATFORM_POSIX

#endif  // ONEFLOW_CORE_COMM_NETWORK_IBVERBS_IBVERBS_QP_H_
