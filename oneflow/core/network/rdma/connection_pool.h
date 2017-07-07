#ifndef ONEFLOW_CORE_NETWORK_RDMA_CONNECTION_POOL_H_
#define ONEFLOW_CORE_NETWORK_RDMA_CONNECTION_POOL_H_

#include <unordered_map>
#include "oneflow/core/network/rdma/switch.h"

namespace oneflow {

class ConnectionPool {
 public:
  ConnectionPool();
  ~ConnectionPool();

  void AddConnection(int64_t peer_machine_id, Connection* conn);
  void CleanConnection(int64_t peer_machine_id);
  Connection* GetConnection(int64_t peer_machine_id) const;

 private:
  int32_t conn_num_;
  std::unordered_map<int64_t, Connection*> connection_dict_;
  ConnectionPool(const ConnectionPool& other) = delete;
  Connection& operator=(const ConnectionPool& other) = delete;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NETWORK_RDMA_CONNECTION_POOL_H_
