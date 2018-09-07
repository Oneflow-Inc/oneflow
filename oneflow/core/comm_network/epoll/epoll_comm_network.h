#ifndef ONEFLOW_CORE_COMM_NETWORK_EPOLL_EPOLL_COMM_NETWORK_H_
#define ONEFLOW_CORE_COMM_NETWORK_EPOLL_EPOLL_COMM_NETWORK_H_

#include "oneflow/core/comm_network/comm_network.h"
#include "oneflow/core/comm_network/epoll/socket_helper.h"
#include "oneflow/core/comm_network/epoll/socket_memory_desc.h"

#ifdef PLATFORM_POSIX

namespace oneflow {

class EpollCommNet final : public CommNetIf<SocketMemDesc> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EpollCommNet);
  ~EpollCommNet();

  static void Init(const Plan& plan) { Global<CommNet>::SetAllocated(new EpollCommNet(plan)); }

  void RegisterMemoryDone() override;

  void SendActorMsg(int64_t dst_machine_id, const ActorMsg& msg) override;
  void SendSocketMsg(int64_t dst_machine_id, const SocketMsg& msg);

 private:
  SocketMemDesc* NewMemDesc(void* ptr, size_t byte_size) override;

  EpollCommNet(const Plan& plan);
  void InitSockets();
  SocketHelper* GetSocketHelper(int64_t machine_id);
  void DoRead(void* read_id, int64_t src_machine_id, void* src_token, void* dst_token) override;

  std::vector<IOEventPoller*> pollers_;
  std::vector<int> machine_id2sockfd_;
  HashMap<int, SocketHelper*> sockfd2helper_;
};

template<>
class Global<EpollCommNet> final {
 public:
  static EpollCommNet* Get() { return static_cast<EpollCommNet*>(Global<CommNet>::Get()); }
};

}  // namespace oneflow

#endif  // PLATFORM_POSIX

#endif  // ONEFLOW_CORE_COMM_NETWORK_EPOLL_EPOLL_COMM_NETWORK_H_
