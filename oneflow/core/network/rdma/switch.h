#ifndef ONEFLOW_CORE_NETWORK_RDMA_SWITCH_H_
#define ONEFLOW_CORE_NETWORK_RDMA_SWITCH_H_

#ifdef WIN32
#include "oneflow/core/network/rdma/netdirect/interface.h"
#else
#include "oneflow/core/network/rdma/verbs/interface.h"
#endif

#endif  // ONEFLOW_CORE_NETWORK_RDMA_SWITCH_H_
