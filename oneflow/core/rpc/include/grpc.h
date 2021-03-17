#ifndef ONEFLOW_CORE_RPC_INCLUDE_GRPC_H_
#define ONEFLOW_CORE_RPC_INCLUDE_GRPC_H_

#include "oneflow/core/rpc/include/base.h"

namespace oneflow {

class GrpcRpcManager : public RpcManager {
 public:
  GrpcRpcManager() {}
  ~GrpcRpcManager();
  Maybe<void> Bootstrap();
  Maybe<void> CreateServer();
  Maybe<void> CreateClient();
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_RPC_INCLUDE_GRPC_H_
