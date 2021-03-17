#ifndef ONEFLOW_CORE_RPC_INCLUDE_GRPC_H_
#define ONEFLOW_CORE_RPC_INCLUDE_GRPC_H_

#include "oneflow/core/rpc/include/base.h"
#include "oneflow/core/control/ctrl_bootstrap.pb.h"

namespace oneflow {

class GrpcCtrlClient final : public CtrlClient {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GrpcCtrlClient);
  GrpcCtrlClient(const ProcessCtx& process_ctx);
  ~GrpcCtrlClient();

  void Barrier(const std::string& barrier_name);
  void Barrier(const std::string& barrier_name, int32_t barrier_num);

  TryLockResult TryLock(const std::string& name);
  void NotifyDone(const std::string& name);
  void WaitUntilDone(const std::string& name);

  void PushKV(const std::string& k, std::function<void(std::string*)> VSetter);
  void PushKV(const std::string& k, const std::string& v);
  void PushKV(const std::string& k, const PbMessage& msg);
  void PushMasterKV(const std::string& k, const PbMessage& msg);

  void ClearKV(const std::string& k);
  void ClearMasterKV(const std::string& k);

  void PullKV(const std::string& k, std::function<void(const std::string&)> VGetter);
  void PullKV(const std::string& k, std::string* v);
  void PullKV(const std::string& k, PbMessage* msg);
  void PullMasterKV(const std::string& k, PbMessage* msg);
  void PushActEvent(const ActEvent&){};
  void Clear();

  int32_t IncreaseCount(const std::string& k, int32_t v);
  int32_t IncreaseCount(const std::string& k) { return IncreaseCount(k, 1); }
  void EraseCount(const std::string& k);

 protected:
  void PushMasterKV(const std::string& k, std::function<void(std::string*)> VSetter);
  void PullMasterKV(const std::string& k, std::function<void(const std::string&)> VGetter);

 private:
  const ProcessCtx& process_ctx() const { return process_ctx_; }
  ProcessCtx process_ctx_;
  bool need_heartbeat_thread_stop_;
  std::mutex need_heartbeat_thread_stop_mtx_;
  std::thread heartbeat_thread_;
  RpcClient rpc_client_;
};

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
