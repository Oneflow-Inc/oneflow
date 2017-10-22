#include "oneflow/core/comm_network/rdma/connection.h"

namespace oneflow {

Connection::Connection() : qp_(nullptr) { conn_ = new Connector(); }

Connection::~Connection() {
  if (conn_ != nullptr) { delete conn_; }
  if (qp_ != nullptr) { CHECK_EQ(ibv_destroy_qp(qp_), 0); }
}

void Connection::ConnectTo(int64_t peer_machine_id) {
  ConnectorInfo peer_conn_info =
      CtrlClient::Singleton()->PullConnectorInfo(peer_machine_id);
  conn_->peer_conn_info.lid = peer_conn_info.lid;
  conn_->peer_conn_info.qpn = peer_conn_info.qpn;
  conn_->peer_conn_info.snp = peer_conn_info.snp;
  conn_->peer_conn_info.iid = peer_conn_info.iid;
}

}  // namespace oneflow
