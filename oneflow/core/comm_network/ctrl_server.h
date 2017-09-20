#ifndef ONEFLOW_CORE_COMM_NETWORK_CTRL_SERVER_H_
#define ONEFLOW_CORE_COMM_NETWORK_CTRL_SERVER_H_

#include "grpc++/server_builder.h"
#include "oneflow/core/comm_network/ctrl_call.h"

namespace oneflow {

class CtrlServer final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CtrlServer);
  CtrlServer() = delete;
  ~CtrlServer();

  CtrlServer(const std::string& server_addr);

 private:
  void HandleRpcs();

#define DECLARE_CTRL_METHOD_HANDLE(method) \
  void method##Handler(CtrlCallIf* call);

  OF_PP_FOR_EACH_TUPLE(DECLARE_CTRL_METHOD_HANDLE, CTRL_METHOD_SEQ);

#undef DECLARE_CTRL_METHOD_HANDLE

  std::unique_ptr<CtrlService::AsyncService> grpc_service_;
  std::unique_ptr<grpc::ServerCompletionQueue> cq_;
  std::unique_ptr<grpc::Server> grpc_server_;
  std::list<CtrlCallIf*> added_worker_calls_;
  std::thread loop_thread_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMM_NETWORK_CTRL_SERVER_H_
