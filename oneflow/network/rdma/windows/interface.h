#ifndef ONEFLOW_NETWORK_RDMA_WINDOWS_INTERFACE_H_
#define ONEFLOW_NETWORK_RDMA_WINDOWS_INTERFACE_H_

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include "ndsupport.h"

struct Connection {
    int32_t peer_rank = { -1 };

    IND2Connector* connector;
    IND2QueuePair* queue_pair;
    
    OVERLAPPED ov;
};

#endif // ONEFLOW_NETWORK_RDMA_WINDOWS_INTERFACE_H_
