#ifndef ONEFLOW_NETWORK_RDMA_CONNECTION_POOL_H_
#define ONEFLOW_NETWORK_RDMA_CONNECTION_POOL_H_

#include <unordered_map>

#include "network/rdma/agency.h"

namespace oneflow {

class ConnectionPool {
 public:
  ConnectionPool() = default;
  ~ConnectionPool() = default;

  void AddConnection(uint64_t peer_machine_id, Connection* conn);
  void CleanConnection(uint64_t peer_machine_id);
  Connection* GetConnection(uint64_t peer_machine_id) const;

 private:
  int32_t conn_num_;
  std::unordered_map<uint64_t, Connection*> connection_dict_;
  ConnectionPool(const ConnectionPool& other) = delete;
  Connection& operator=(const ConnectionPool& other) = delete;
};

}  // namespace oneflow

#endif  // ONEFLOW_NETWORK_RDMA_CONNECTION_POOL_H_
