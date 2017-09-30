#ifndef ONEFLOW_CORE_COMM_NETWORK_RPC_CTRL_SERVER_H_
#define ONEFLOW_CORE_COMM_NETWORK_RPC_CTRL_SERVER_H_

#include "grpc++/server_builder.h"
#include "oneflow/core/comm_network/rpc/ctrl_call.h"

namespace oneflow {

class CtrlServer final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CtrlServer);
  CtrlServer() = delete;
  ~CtrlServer();

  CtrlServer(const std::string& server_addr);
  void PublishPlan(const Plan* plan);

 private:
  void HandleRpcs();

#define DECLARE_CTRL_METHOD_HANDLE(method) \
  void method##Handler(CtrlCall<method##Request, method##Response>* call);

  OF_PP_FOR_EACH_TUPLE(DECLARE_CTRL_METHOD_HANDLE, CTRL_METHOD_SEQ);

#undef DECLARE_CTRL_METHOD_HANDLE

  std::unique_ptr<CtrlService::AsyncService> grpc_service_;
  std::unique_ptr<grpc::ServerCompletionQueue> cq_;
  std::unique_ptr<grpc::Server> grpc_server_;
  std::thread loop_thread_;
  // AddWorker
  std::list<CtrlCallIf*> added_worker_calls_;
  // Barrier
  HashMap<std::string, std::pair<std::list<CtrlCallIf*>, int32_t>>
      barrier_calls_;
  // TryLock, NotifyDone, WaitUntilDone
  HashMap<std::string, void*> name2lock_status_;  // TODO: erase outdated item
  // FetchPlan
  std::mutex plan_mtx_;
  const Plan* plan_;
  std::list<CtrlCallIf*> pending_plan_calls_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMM_NETWORK_RPC_CTRL_SERVER_H_
