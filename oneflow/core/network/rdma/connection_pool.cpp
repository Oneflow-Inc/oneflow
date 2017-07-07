#include "oneflow/core/network/rdma/connection_pool.h"

#include <cstdint>
#include <memory>

namespace oneflow {

ConnectionPool::ConnectionPool() : conn_num_(0) {}

ConnectionPool::~ConnectionPool() {
  connection_dict_.erase (connection_dict_.begin(), conection_dict_.end());
}

void ConnectionPool::AddConnection(int64_t peer_machine_id, Connection* conn) {
  connection_dict_.insert({peer_machine_id, conn});
  conn_num_++;
}

void ConnectionPool::CleanConnection(int64_t peer_machine_id) {
  Connection* conn = GetConnection(peer_machine_id);
  if (conn != nullptr) {
    conn->DestroyConnection();
    delete conn;
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
