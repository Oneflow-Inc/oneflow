#include "network/rdma/connection_pool.h"

namespace oneflow {

void ConnectionPool::AddConnection(int32_t peer_rank, Connection* conn)
{
    conn->BuildConnection(peer_rank);
    connection_dict_.insert({ peer_rank, conn });
    connNum++;
}

void ConnectionPool::CleanConnection(int32_t peer_rank)
{
    Connection* conn = GetConnection(peer_rank);
    if (conn != nullptr) {
        conn->DestroyConnection();
        delete conn;
        connNum--;
    }
}

Connection* ConnectionPool::GetConnection(peer_rank) const
{
    auto conn_it = connection_dict_.find(peer_rank);
    if (conn_it != connection_dict_.end()) {
        Connection* conn = conn_it->second;
        return conn;
    }
    else {
        return NULL;
    }
}

} // namespace oneflow
