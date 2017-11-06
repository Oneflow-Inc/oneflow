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
  void PostNewWSARecv2Socket(int64_t dst_machine_id);

  // On...Done() will change the IOData* content according to some conditions
  void OnRecvMsgHeadDone(IOData* io_data_ptr);
  void OnRecvMsgBodyDone(IOData* io_data_ptr);
  void OnSendMsgHeadDone(IOData* io_data_ptr);
  void OnSendDone(IOData* io_data_ptr);
  // when post send msg request to completion port, need check send queue for
  // send order
  void OnFirstSendMsgHead(IOData* io_data_ptr);

  // reset a IOData->data_buff to this IOData->SocketMsg
  void ResetIODataBuff(IOData* io_data_ptr);

  void WSARecvFromIOData(IOData* io_data_ptr);
  void WSASendFromIOData(IOData* io_data_ptr);

  std::vector<SOCKET> machine_id2socket_;
  std::vector<IOData*> machine_id2io_data_recv_;
  std::vector<std::mutex> machine_id2send_que_mtx_;
  std::vector<std::queue<IOData*>> machine_id2io_data_send_que_;

  HANDLE completion_port_;
  int32_t num_of_concurrent_threads_;
  int64_t this_machine_id_;
  int64_t total_machine_num_;
};

}  // namespace oneflow

#endif  // PLATFORM_WINDOW

#endif  // ONEFLOW_CORE_COMM_NETWORK_IOCP_IO_WORKER_H_
