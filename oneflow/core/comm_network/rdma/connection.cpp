#include "oneflow/core/comm_network/rdma/connection.h"

namespace oneflow {

Connection::Connection() : qp_(nullptr) { conn_ = new Connector(); }

Connection::~Connection() {
  if (conn_ != nullptr) { delete conn_; }
  if (qp_ != nullptr) { CHECK_EQ(ibv_destroy_qp(qp_), 0); }
}

}  // namespace oneflow
