#ifndef ONEFLOW_CORE_COMM_NETWORK_RDMA_CONNECTION_POOL_H_
#define ONEFLOW_CORE_COMM_NETWORK_RDMA_CONNECTION_POOL_H_

#include "oneflow/core/comm_network/rdma/connection.h"
#include "oneflow/core/comm_network/rdma/rdma_memory.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

class ConnectionPool {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConnectionPool);
  ConnectionPool();
  ~ConnectionPool();

  void AddConnection(int64_t peer_machine_id, Connection* conn);
  void CleanConnection(int64_t peer_machine_id);
  Connection* GetConnection(int64_t peer_machine_id) const;

 private:
  int32_t conn_num_;
  std::unordered_map<int64_t, Connection*> conn_dict_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMM_NETWORK_RDMA_CONNECTION_POOL_H_
