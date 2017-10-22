#include "oneflow/core/comm_network/rdma/connection_pool.h"

namespace oneflow {

ConnectionPool::ConnectionPool() : conn_num_(0) {}

ConnectionPool::~ConnectionPool() {
  for (auto& pair : conn_dict_) { CleanConnection(pair.first); }
  conn_dict_.clear();
}

void ConnectionPool::AddConnection(int64_t peer_machine_id, Connection* conn) {
  conn_dict_.insert({peer_machine_id, conn});
  conn_num_++;
}

void ConnectionPool::CleanConnection(int64_t peer_machine_id) {
  Connection* conn = GetConnection(peer_machine_id);
  if (conn != nullptr) {
    delete conn;
    conn = nullptr;
    conn_num_--;
  }
}

Connection* ConnectionPool::GetConnection(int64_t peer_machine_id) const {
  auto conn_it = conn_dict_.find(peer_machine_id);
  if (conn_it != conn_dict_.end()) {
    return conn_it->second;
  } else {
    return nullptr;
  }
}

}  // namespace oneflow
