#ifndef ONEFLOW_CORE_RPC_INCLUDE_MANAGER_H_
#define ONEFLOW_CORE_RPC_INCLUDE_MANAGER_H_

namespace oneflow {

class RpcManager {
 public:
  RpcManager() {}
  virtual ~RpcManager() {}
  virtual void CreateServer() {}
  virtual void CreateClient() {}
  virtual void Bootstrap() {}
  virtual void TearDown() {}
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_RPC_INCLUDE_MANAGER_H_
