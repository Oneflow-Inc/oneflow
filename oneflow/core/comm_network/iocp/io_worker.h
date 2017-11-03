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

  void PostSendMsgRequest(int64_t dst_machine_id, SocketMsg socket_msg);
  void Start();
  void Stop();
 private:
  static DWORD WINAPI StartThreadProc(LPVOID pParam) {
    IOWorker* this_worker = static_cast<IOWorker*>(pParam);
    return this_worker->ThreadProc();
  }
  DWORD ThreadProc();

  void InitSockets();
  void PostWSARecv2Socket();

  std::vector<SOCKET> machine_id2socket_;
  std::vector<IOData*> machine_id2io_data_recv_;
  std::mutex send_que_mtx_;
  std::vector<std::queue<IOData*>> machine_id2io_data_send_que_;

  HANDLE completion_port_;
  int32_t num_of_concurrent_threads_;
  int64_t this_machine_id_;
  int64_t total_machine_num_;
};

}  // namespace oneflow

#endif  // PLATFORM_WINDOW

#endif  //ONEFLOW_CORE_COMM_NETWORK_IOCP_IO_WORKER_H_
