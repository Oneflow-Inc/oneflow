#ifndef ONEFLOW_NETWORK_RDMA_AGENCY_H_
#define ONEFLOW_NETWORK_RDMA_AGENCY_H_

#ifdef WIN32
#include "network/rdma/windows/interface.h"
#else
#include "network/rdma/linux/interface.h"
#endif

#endif  // ONEFLOW_NETWORK_RDMA_AGENCY_H_
