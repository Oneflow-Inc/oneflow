#include "oneflow/core/network/rdma/connection_pool.h"

namespace oneflow {

ConnectionPool::ConnectionPool() : conn_num_(0) {}

ConnectionPool::~ConnectionPool() {
  for (auto& pair : connection_dict_) {
    CleanConnection(pair.first);
  }
  connection_dict_.clear();
}

void ConnectionPool::AddConnection(int64_t peer_machine_id, Connection* conn) {
  connection_dict_.insert({peer_machine_id, conn});
  conn_num_++;
}

void ConnectionPool::CleanConnection(int64_t peer_machine_id) {
  Connection* conn = GetConnection(peer_machine_id);
  if (conn != nullptr) {
    conn->Destroy();
    delete conn;
    conn = nullptr;
    conn_num_--;
  }
}

Connection* ConnectionPool::GetConnection(int64_t peer_machine_id) const {
  auto conn_it = connection_dict_.find(peer_machine_id);
  if (conn_it != connection_dict_.end()) {
    Connection* conn = conn_it->second;
    return conn;
  } else {
    return nullptr;
  }
}

}  // namespace oneflow
