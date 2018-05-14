#include "oneflow/core/comm_network/ibverbs/ibverbs_comm_network.h"
#include "oneflow/core/control/ctrl_client.h"

#if defined(WITH_RDMA) && defined(PLATFORM_POSIX)

namespace oneflow {

namespace {

std::string GenTokensMsgKey(int64_t machine_id) {
  return "IBVerbsTokensMsg/" + std::to_string(machine_id);
}

std::string GenConnInfoKey(int64_t src_machine_id, int64_t dst_machine_id) {
  return "IBVerbsConnInfo/" + std::to_string(src_machine_id) + " " + std::to_string(dst_machine_id);
}

}  // namespace

IBVerbsCommNet::~IBVerbsCommNet() { TODO(); }

const void* IBVerbsCommNet::RegisterMemory(void* mem_ptr, size_t byte_size) {
  IBVerbsMemDesc* mem_desc = new IBVerbsMemDesc(pd_, mem_ptr, byte_size);
  mem_desc_mgr_.RegisterMemDesc(mem_desc);
  return mem_desc;
}

void IBVerbsCommNet::UnRegisterMemory(const void* token) { mem_desc_mgr_.UnRegisterMemDesc(); }

void IBVerbsCommNet::RegisterMemoryDone() {
  int64_t total_machine_num = Global<JobDesc>::Get()->TotalMachineNum();
  int64_t this_machine_id = Global<MachineCtx>::Get()->this_machine_id();
  IBVerbsTokensMsg this_tokens_msg;
  for (IBVerbsMemDesc* mem_desc : mem_desc_mgr_.mem_descs()) {
    this_tokens_msg.mutable_token2mem_desc()->insert(
        {reinterpret_cast<uint64_t>(mem_desc), mem_desc->ToProto()});
  }
  Global<CtrlClient>::Get()->PushKV(GenTokensMsgKey(this_machine_id), this_tokens_msg);
  FOR_RANGE(int64_t, peer_machine_id, 0, total_machine_num) {
    if (peer_machine_id == this_machine_id) { continue; }
    IBVerbsTokensMsg peer_tokens_msg;
    Global<CtrlClient>::Get()->PullKV(GenTokensMsgKey(peer_machine_id), &peer_tokens_msg);
    for (const auto& pair : peer_tokens_msg.token2mem_desc()) {
      CHECK(token2mem_desc_.insert({reinterpret_cast<void*>(pair.first), pair.second}).second);
    }
  }
  OF_BARRIER();
  Global<CtrlClient>::Get()->ClearKV(GenTokensMsgKey(this_machine_id));
}

void IBVerbsCommNet::SendActorMsg(int64_t dst_machine_id, const ActorMsg& msg) {
  qp_vec_.at(dst_machine_id)->PostSendRequest(msg);
}

IBVerbsCommNet::IBVerbsCommNet(const Plan& plan) {
  GenConnectionInfo(plan);
  // Init Adapter
  ibv_device** device_list = ibv_get_device_list(NULL);
  ibv_device* device = device_list[0];
  context_ = ibv_open_device(device);
  CHECK(context_);
  pd_ = ibv_alloc_pd(context_);
  CHECK(pd_);

  // Init env
  send_cq_ = ibv_create_cq(context_, 10, NULL, NULL, 0);  // cqe
  CHECK(send_cq_);
  recv_cq_ = ibv_create_cq(context_, 10, NULL, NULL, 0);  // cqe
  CHECK(recv_cq_);

  ibv_free_device_list(device_list);
  InitRdma();
  Start();
  int64_t this_machine_id = Global<MachineCtx>::Get()->this_machine_id();
  for (int64_t peer_machine_id : Global<CommNet>::Get()->peer_machine_id()) {
    IBVerbsConnection* conn = NewIBVerbsConnection();
    connection_pool_.emplace(peer_machine_id, conn);
    Global<CtrlClient>::Get()->PushKV(GenConnInfoKey(this_machine_id, peer_machine_id),
                                      conn->mut_this_machine_conn_info());
  }
  OF_BARRIER();
  for (int64_t peer_machine_id : Global<CommNet>::Get()->peer_machine_id()) {
    IBVerbsConnection* conn = connection_pool_[peer_machine_id];
    Global<CtrlClient>::Get()->PullKV(GenConnInfoKey(peer_machine_id, this_machine_id),
                                      conn->mut_peer_machine_conn_info_ptr());
    for (size_t i = 0; i != kPrePostRecvNum; ++i) {
      ActorMsg* actor_msg = new ActorMsg;
      auto ibverbs_mem_desc = NewIBVerbsMemDesc(actor_msg, sizeof(ActorMsg));
      recv_msg2conn_ptr_.emplace(actor_msg, conn);
      recv_msg2mem_desc_.emplace(actor_msg, ibverbs_mem_desc);
      conn->PostRecvRequest(actor_msg, ibverbs_mem_desc);
    }
    conn->CompleteConnection();
  }
  OF_BARRIER();
}

void IBVerbsCommNet::DoRead(ReadContext* read_ctx, int64_t src_machine_id, const void* src_token,
                            const void* dst_token) {
  qp_vec_.at(src_machine_id)
      ->PostReadRequest(token2mem_desc_.at(src_token),
                        static_cast<const IBVerbsMemDesc*>(dst_token), read_ctx);
}

void PollCQ() {
  ibv_wc wc;
  int32_t num_comp = ibv_poll_cq(send_cq_, 1, &wc);
  CHECK_GE(num_comp, 0);
  if (num_comp == 0) { return; }

  if (wc.status != IBV_WC_SUCCESS) { LOG(FATAL) << "PollSendQueue Error Code: " << wc.status; }
  switch (wc.opcode) {
    case IBV_WC_SEND: {
      ReleaseSendMsg(reinterpret_cast<ActorMsg*>(wc.wr_id));
      return;
    }
    case IBV_WC_RDMA_READ: {
      Global<CommNet>::Get()->ReadDone(reinterpret_cast<void*>(wc.wr_id));
      return;
    }
    default: return;
  }
  ibv_wc wc;
  int32_t num_comp = ibv_poll_cq(recv_cq_, 1, &wc);
  CHECK_GE(num_comp, 0);
  if (num_comp == 0) { return; }

  if (wc.status != IBV_WC_SUCCESS) { LOG(FATAL) << "PollRecvQueue Error Code:  " << wc.status; }
  ActorMsg* msg = reinterpret_cast<ActorMsg*>(wc.wr_id);

  Global<ActorMsgBus>::Get()->SendMsg(*msg);
  CHECK(recv_msg2conn_ptr_.find(msg) != recv_msg2conn_ptr_.end());
  IBVerbsConnection* conn = recv_msg2conn_ptr_.at(msg);
  auto msg2mem_it = recv_msg2mem_desc_.find(msg);
  conn->PostRecvRequest(msg, msg2mem_it->second);
}

}  // namespace oneflow

#endif  // WITH_RDMA && PLATFORM_POSIX
