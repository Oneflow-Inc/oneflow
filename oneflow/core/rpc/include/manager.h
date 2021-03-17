#ifndef ONEFLOW_CORE_RPC_INCLUDE_MANAGER_H_
#define ONEFLOW_CORE_RPC_INCLUDE_MANAGER_H_

#ifdef RPC_BACKEND_GRPC
#include "oneflow/core/rpc/include/grpc.h"
#endif  // RPC_BACKEND_GRPC

#ifdef RPC_BACKEND_LOCAL
#include "oneflow/core/rpc/include/local.h"
#endif  // RPC_BACKEND_LOCAL

#endif  // ONEFLOW_CORE_RPC_INCLUDE_MANAGER_H_
