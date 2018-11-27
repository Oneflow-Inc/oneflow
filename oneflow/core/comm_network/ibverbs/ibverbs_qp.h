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
  ActorMsgMR(ibv_pd* pd) {
    mem_desc_.reset(new IBVerbsMemDesc(pd, &msg_, sizeof(msg_)));
    CHECK_EQ(mem_desc_->sge_vec().size(), 1);
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

class IBVerbsQP final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IBVerbsQP);
  IBVerbsQP() = delete;
  IBVerbsQP(ibv_context*, ibv_pd*, ibv_cq*);
  ~IBVerbsQP();

  uint32_t qp_num() const { return qp_->qp_num; }
  void Connect(const IBVerbsConnectionInfo& peer_info);
  void PostAllRecvRequest();

  void PostReadRequest(const IBVerbsMemDescProto& remote_mem, const IBVerbsMemDesc& local_mem,
                       void* read_id);
  void PostSendRequest(const ActorMsg& msg);

  void ReadDone(WorkRequestId*);
  void SendDone(WorkRequestId*);
  void RecvDone(WorkRequestId*);

 private:
  WorkRequestId* NewWorkRequestId();
  void DeleteWorkRequestId(WorkRequestId* wr_id);
  ActorMsgMR* GetOneSendMsgMRFromBuf();
  void PostRecvRequest(ActorMsgMR*);

  const IBVerbsConf& ibv_conf_;
  ibv_context* ctx_;
  ibv_pd* pd_;
  ibv_qp* qp_;
  std::vector<ActorMsgMR*> recv_msg_buf_;

  std::mutex send_msg_buf_mtx_;
  std::queue<ActorMsgMR*> send_msg_buf_;
};

}  // namespace oneflow

#endif  // WITH_RDMA && PLATFORM_POSIX

#endif  // ONEFLOW_CORE_COMM_NETWORK_IBVERBS_IBVERBS_QP_H_
