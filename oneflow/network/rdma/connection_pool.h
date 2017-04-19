#ifndef ONEFLOW_NETWORK_RDMA_CONNECTION_POOL_H_
#define ONEFLOW_NETWORK_RDMA_CONNECTION_POOL_H_

#include <unordered_map>
#include <cstdint>
#include <memory>
#include "network/rdma/switch.h"

namespace oneflow {

class ConnectionPool {
public:
    ConnectionPool() = default;
    ~ConnectionPool() = default;

    void AddConnection(int32_t peer_rank, Connection* conn);
    void CleanConnection(int32_t peer_rank);
    Connection* GetConnection(int32_t peer_rank) const;

private:
    int32_t connNum_;
    std::unordered_map<int32_t, Connection*> connection_dict_;
    ConnectionPool(const ConnectionPool& other) = delete;
    Connection& operator=(const ConnectionPool& other) = delete;
};

} // namespace oneflow

#endif // ONEFLOW_NETWORK_RDMA_CONNECTION_POOL_H_
