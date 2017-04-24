#ifndef ONEFLOW_NETWORK_RDMA_WINDOWS_CONNECTION_H_
#define ONEFLOW_NETWORK_RDMA_WINDOWS_CONNECTION_H_

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
// #include <unistd.h>
#include <cstdint>
#include "network/rdma/windows/ndsupport.h"

namespace oneflow {

class Connection {
public: 
    Connection();
    ~Connection();
    Connection(uint64_t peer_machine_id);

private:
    uint64_t peer_machine_id_ = { -1 };

    IND2Connector* connector_;
    IND2QueuePair* queue_pair_;
    OVERLAPPED ov_;

    //map peer_rank to id

    //void BuildConnection();
    //void BuildContext();
    //void BuildParams();
    ///
    //
};

} // namespace oneflow

#endif // ONEFLOW_NETWORK_RDMA_WINDOWS_CONNECTION_H_
