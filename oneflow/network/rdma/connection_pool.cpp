#include "network/rdma/connection_pool.h"

#include <cstdint>
#include <memory>
#include <unordered_map>

namespace oneflow {

void ConnectionPool::AddConnection(uint64_t peer_machine_id, 
                                   Connection* conn) {
  conn->BuildConnection(peer_machine_id);
  connection_dict_.insert({ peer_machine_id, conn });
  conn_num_++;
}

void ConnectionPool::CleanConnection(uint64_t peer_machine_id) {
  Connection* conn = GetConnection(peer_machine_id);
  if (conn != nullptr) {
    conn->DestroyConnection();
    delete conn;
    conn_num_--;
  }
}

Connection* ConnectionPool::GetConnection(uint64_t peer_machine_id) const {
  auto conn_it = connection_dict_.find(peer_machine_id);
  if (conn_it != connection_dict_.end()) {
    Connection* conn = conn_it->second;
    return conn;
  }
  else {
     return NULL;
  }
	//return nullptr;
}

} // namespace oneflow
