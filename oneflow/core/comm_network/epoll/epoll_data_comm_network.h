#ifndef ONEFLOW_CORE_COMM_NETWORK_POSIX_POSIX_DATA_COMM_NETWORK_H_
#define ONEFLOW_CORE_COMM_NETWORK_POSIX_POSIX_DATA_COMM_NETWORK_H_

#include "oneflow/core/comm_network/data_comm_network.h"

#ifdef PLATFORM_POSIX

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/ip.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <sys/types.h>

namespace oneflow {

class EpollDataCommNet final : public DataCommNet {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EpollDataCommNet);
  EpollDataCommNet() = delete;
  ~EpollDataCommNet() = default;

  static void Init(const Plan& plan);

  const void* RegisterMemory(void* mem_ptr, size_t byte_size) override;
  void UnRegisterMemory(const void* token) override;
  void RegisterMemoryDone() override;

  void* CreateStream() override;
  void Read(void* stream_id, const void* src_token,
            const void* dst_token) override;
  void AddCallBack(void* stream_id, std::function<void()>) override;

  void SendActorMsg(int64_t dst_machine_id, const ActorMsg& msg) override;

 private:
  EpollDataCommNet(const Plan& plan);

  // Memory Desc
  struct MemDesc {
    void* mem_ptr;
    size_t byte_size;
  };
  std::mutex mem_desc_mtx_;
  std::list<MemDesc*> mem_descs_;
  size_t unregister_mem_descs_cnt_;
  // socket fd
  std::vector<int> socket_fds_;
};

}  // namespace oneflow

#endif  // PLATFORM_POSIX

#endif  // ONEFLOW_CORE_COMM_NETWORK_POSIX_POSIX_DATA_COMM_NETWORK_H_
