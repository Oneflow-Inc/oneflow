#ifndef ONEFLOW_CORE_COMM_NETWORK_EPOLL_EPOLL_DATA_COMM_NETWORK_H_
#define ONEFLOW_CORE_COMM_NETWORK_EPOLL_EPOLL_DATA_COMM_NETWORK_H_

#include "oneflow/core/comm_network/data_comm_network.h"
#include "oneflow/core/comm_network/epoll/socket_helper.h"
#include "oneflow/core/comm_network/epoll/socket_memory_desc.h"

#ifdef PLATFORM_POSIX

namespace oneflow {

class EpollDataCommNet final : public DataCommNet {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EpollDataCommNet);
  EpollDataCommNet() = delete;
  ~EpollDataCommNet();

  static EpollDataCommNet* Singleton() {
    return static_cast<EpollDataCommNet*>(DataCommNet::Singleton());
  }

  static void Init(uint16_t port);

  const void* RegisterMemory(void* mem_ptr, size_t byte_size) override;
  void UnRegisterMemory(const void* token) override;
  void RegisterMemoryDone() override;

  void* Read(int64_t src_machine_id, const void* src_token,
             const void* dst_token) override;
  void AddReadCallBack(void* read_id, std::function<void()> callback) override;

  void SendActorMsg(int64_t dst_machine_id, const ActorMsg& msg) override;
  void SendSocketMsg(int64_t dst_machine_id, const SocketMsg& msg);

 private:
  EpollDataCommNet(uint16_t port);
  void InitSockets(uint16_t port);
  SocketHelper* GetSocketHelper(int64_t machine_id);

  // Memory Desc
  std::mutex mem_desc_mtx_;
  std::list<SocketMemDesc*> mem_descs_;
  size_t unregister_mem_descs_cnt_;
  // Socket
  std::vector<IOEventPoller*> pollers_;
  std::vector<int> machine_id2sockfd_;
  HashMap<int, SocketHelper*> sockfd2helper_;
};

}  // namespace oneflow

#endif  // PLATFORM_POSIX

#endif  // ONEFLOW_CORE_COMM_NETWORK_EPOLL_EPOLL_DATA_COMM_NETWORK_H_
