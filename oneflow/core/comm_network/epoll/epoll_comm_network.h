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
  void RequestRead(int64_t dst_machine_id, void* src_token, void* dst_token, void* read_id);
  void PartReadDone(void* read_id, void* dst_token, int32_t part_num);

 private:
  SocketMemDesc* NewMemDesc(void* ptr, size_t byte_size) override;

  EpollCommNet(const Plan& plan);
  void InitSockets();
  void SetSocketHelper(int64_t machine_id, int32_t sockfd);
  SocketHelper* GetSocketHelper(int64_t machine_id, int32_t link_index) const;
  void DoRead(void* read_id, int64_t src_machine_id, void* src_token, void* dst_token) override;

  const EpollConf& epoll_conf_;
  size_t poller_idx_;
  std::vector<IOEventPoller*> pollers_;
  HashMap<int64_t, std::vector<int32_t>> machine_id2sockfds_;
  HashMap<int32_t, SocketHelper*> sockfd2helper_;
  HashMap<void*, std::atomic<int32_t>> dst_token2part_done_cnt_;
};

template<>
class Global<EpollCommNet> final {
 public:
  static EpollCommNet* Get() { return static_cast<EpollCommNet*>(Global<CommNet>::Get()); }
};

}  // namespace oneflow

#endif  // PLATFORM_POSIX

#endif  // ONEFLOW_CORE_COMM_NETWORK_EPOLL_EPOLL_COMM_NETWORK_H_
