#include "oneflow/core/comm_network/rdma/connection.h"

namespace oneflow {

Connection::Connection() : qp_ptr_(nullptr) {}

Connection::~Connection() {
  if (qp_ptr_ != nullptr) { CHECK_EQ(ibv_destroy_qp(qp_ptr_), 0); }
}

void Connection::PostReadRequest(void* read_ctx, const RdmaMem* local_mem,
                                 const RdmaMemDesc* remote_mem) {
  // TODO
}

void Connection::PostSendRequest(const RdmaMem* msg_mem) {
  // TODO
}

void Connection::PostRecvRequest(const RdmaMem* msg_mem) {
  // TODO
}

void Connection::WaitForConnection() {
  // TODO
}

}  // namespace oneflow
