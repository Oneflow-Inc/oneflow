#include "oneflow/core/comm_network/ibverbs/ibverbs_comm_network.h"
#include "oneflow/core/comm_network/ibverbs/ibverbs_tokens_message.pb.h"

#if defined(WITH_RDMA) && defined(PLATFORM_POSIX)

namespace oneflow {

namespace {

std::string GenTokensMsgKey(int64_t machine_id) {
  return "IBVerbsTokensMsg/" + std::to_string(machine_id);
}

}  // namespace

void IBVerbsCommNet::Init() {
  CommNet::Singleton()->set_comm_network_ptr(new IBVerbsCommNet());
}

IBVerbsCommNet::IBVerbsCommNet() { endpoint_manager_ = new EndpointManager(); }

IBVerbsCommNet::~IBVerbsCommNet() { delete endpoint_manager_; }

const void* IBVerbsCommNet::RegisterMemory(void* mem_ptr, size_t byte_size) {
  IBVerbsMemDesc* ibverbs_mem_desc =
      endpoint_manager_->NewIBVerbsMemDesc(mem_ptr, byte_size);
  mem_desc_mgr_.RegisterMemDesc(ibverbs_mem_desc);
  return ibverbs_mem_desc;
}

void IBVerbsCommNet::UnRegisterMemory(const void* token) {
  mem_desc_mgr_.UnRegisterMemDesc();
}

void IBVerbsCommNet::RegisterMemoryDone() {
  int64_t total_machine_num = JobDesc::Singleton()->TotalMachineNum();
  int64_t this_machine_id = MachineCtx::Singleton()->this_machine_id();
  IBVerbsTokensMsg this_machine_tokens_msg;
  const std::list<IBVerbsMemDesc*> mem_descs = mem_desc_mgr_.mem_descs();
  for (auto mem_desc : mem_descs) {
    this_machine_tokens_msg.mutable_token2mem_desc_proto()->insert(
        {reinterpret_cast<uint64_t>(mem_desc),
         mem_desc->IBVerbsMemDescToProto()});
  }
  CtrlClient::Singleton()->PushKV(GenTokensMsgKey(this_machine_id),
                                  this_machine_tokens_msg);
  OF_BARRIER();
  FOR_RANGE(int64_t, peer_machine_id, 0, total_machine_num) {
    if (peer_machine_id == MachineCtx::Singleton()->this_machine_id()) {
      continue;
    }
    IBVerbsTokensMsg peer_machine_tokens_msg;
    CtrlClient::Singleton()->PullKV(GenTokensMsgKey(peer_machine_id),
                                    &peer_machine_tokens_msg);
    for (auto& pair : peer_machine_tokens_msg.token2mem_desc_proto()) {
      CHECK(token2mem_desc_proto_.insert({pair.first, pair.second}).second);
    }
  }
  OF_BARRIER();
}

void* IBVerbsCommNet::Read(void* actor_read_id, int64_t src_machine_id,
                           const void* src_token, const void* dst_token) {
  auto actor_read_ctx = static_cast<ActorReadContext*>(actor_read_id);
  ReadContext* read_ctx = NewReadCtxInActorReadCtx(actor_read_ctx);
  IBVerbsMemDescProto& remote_mem_desc_proto =
      token2mem_desc_proto_[reinterpret_cast<uint64_t>(src_token)];
  auto local_mem_desc = const_cast<IBVerbsMemDesc*>(
      static_cast<const IBVerbsMemDesc*>(dst_token));
  void* read_done_id =
      new std::tuple<ActorReadContext*, ReadContext*>(actor_read_ctx, read_ctx);
  endpoint_manager_->Read(read_done_id, src_machine_id, local_mem_desc,
                          remote_mem_desc_proto);
  return read_ctx;
}

void IBVerbsCommNet::SendActorMsg(int64_t dst_machine_id, const ActorMsg& msg) {
  endpoint_manager_->SendActorMsg(dst_machine_id, msg);
}

}  // namespace oneflow

#endif  // WITH_RDMA && PLATFORM_POSIX
