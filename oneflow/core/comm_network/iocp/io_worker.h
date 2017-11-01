#ifndef ONEFLOW_CORE_COMM_NETWORK_IOCP_IO_WORKER_H_
#define ONEFLOW_CORE_COMM_NETWORK_IOCP_IO_WORKER_H_

#include "oneflow/core/comm_network/iocp/io_type.h"

#ifdef PLATFORM_WINDOWS

namespace oneflow {

class IOWorker final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IOWorker);
  IOWorker();
  ~IOWorker();

  void AddSocket(SOCKET s, int64_t machine_id);
  void PostSendMsgRequest(int64_t dst_machine_id, SocketMsg socket_msg);
  void Start();
  void Stop();
 private:
  DWORD WINAPI IOWorkerThreadProc(LPVOID pParam);
  void InitSockets();

  std::vector<SOCKET> machine_id2socket_;

  std::vector<IOData*> machine_id2io_data_;

  HANDLE completion_port_;
  int32_t num_of_concurrent_threads_;
};

}  // namespace oneflow

#endif  // PLATFORM_WINDOW

#endif  //ONEFLOW_CORE_COMM_NETWORK_IOCP_IO_WORKER_H_
