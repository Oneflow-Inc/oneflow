#ifndef ONEFLOW_CORE_NETWORK_RDMA_AGENCY_H_
#define ONEFLOW_CORE_NETWORK_RDMA_AGENCY_H_

#ifdef WIN32
#include "oneflow/core/network/rdma/windows/interface.h"
#else
#include "oneflow/core/network/rdma/linux/interface.h"
#endif

#endif  // ONEFLOW_NETWORK_RDMA_AGENCY_H_
