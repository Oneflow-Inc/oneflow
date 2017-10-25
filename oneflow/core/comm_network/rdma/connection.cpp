#include "oneflow/core/comm_network/rdma/connection.h"

namespace oneflow {

Connection::Connection() : qp_ptr_(nullptr) {}

Connection::~Connection() {
  if (qp_ptr_ != nullptr) { CHECK_EQ(ibv_destroy_qp(qp_ptr_), 0); }
}

void Connection::PostReadRequest(void* read_ctx, const RdmaMem* local_mem,
                                 const RdmaMemDesc* remote_mem) {
  ibv_send_wr wr, *bad_wr = nullptr;
  wr.wr_id = reinterpret_cast<uint64_t>(read_ctx);
  wr.opcode = IBV_WR_RDMA_READ;
  wr.sg_list = const_cast<RdmaMem*>(local_mem)->ibv_sge_ptr();
  wr.num_sge = 1;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.wr.rdma.remote_addr = remote_mem->mem_ptr();
  wr.wr.rdma.rkey = remote_mem->token();

  ibv_post_send(qp_ptr_, &wr, &bad_wr);
}

void Connection::PostSendRequest(const ActorMsg* msg, const RdmaMem* msg_mem) {
  ibv_send_wr wr, *bad_wr = nullptr;
  wr.wr_id = reinterpret_cast<uint64_t>(msg);
  wr.next = nullptr;
  wr.sg_list = const_cast<RdmaMem*>(msg_mem)->ibv_sge_ptr();
  wr.num_sge = 1;
  wr.opcode = IBV_WR_SEND;
  wr.send_flags = IBV_SEND_SIGNALED;

  ibv_post_send(qp_ptr_, &wr, &bad_wr);
}

void Connection::PostRecvRequest(const ActorMsg* msg, const RdmaMem* msg_mem) {
  ibv_recv_wr wr, *bad_wr = nullptr;
  wr.wr_id = reinterpret_cast<uint64_t>(msg);
  wr.next = nullptr;
  wr.sg_list = const_cast<RdmaMem*>(msg_mem)->ibv_sge_ptr();
  wr.num_sge = 1;

  ibv_post_recv(qp_ptr_, &wr, &bad_wr);
}

void Connection::WaitForConnection() {
  // TODO
}

}  // namespace oneflow
