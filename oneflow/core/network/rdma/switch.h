#ifndef ONEFLOW_CORE_NETWORK_RDMA_SWITCH_H_
#define ONEFLOW_CORE_NETWORK_RDMA_SWITCH_H_

#ifdef WIN32
#include "oneflow/core/network/rdma/windows/interface.h"
#else
#include "oneflow/core/network/rdma/linux/interface.h"
#endif

#endif  // ONEFLOW_CORE_NETWORK_RDMA_SWITCH_H_
