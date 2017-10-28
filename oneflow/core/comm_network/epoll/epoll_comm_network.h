#ifndef ONEFLOW_CORE_COMM_NETWORK_EPOLL_EPOLL_COMM_NETWORK_H_
#define ONEFLOW_CORE_COMM_NETWORK_EPOLL_EPOLL_COMM_NETWORK_H_

#include "oneflow/core/comm_network/comm_network.h"
#include "oneflow/core/comm_network/epoll/socket_helper.h"
#include "oneflow/core/comm_network/epoll/socket_memory_desc.h"

#ifdef PLATFORM_POSIX

namespace oneflow {

class EpollCommNet final : public CommNet {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EpollCommNet);
  ~EpollCommNet();

  static EpollCommNet* Singleton() {
    return static_cast<EpollCommNet*>(CommNet::Singleton());
  }

  static void Init();
  void EstablishNetwork() override{};

  const void* RegisterMemory(void* mem_ptr, size_t byte_size) override;
  void UnRegisterMemory(const void* token) override;
  void RegisterMemoryDone() override;

  void* NewActorReadId();
  void DeleteActorReadId(void* actor_read_id);
  void* Read(void* actor_read_id, int64_t src_machine_id, const void* src_token,
             const void* dst_token) override;
  void AddReadCallBack(void* actor_read_id, void* read_id,
                       std::function<void()> callback) override;
  void AddReadCallBackDone(void* actor_read_id, void* read_id) override;
  void ReadDone(void* read_done_id) override;

  void SendActorMsg(int64_t dst_machine_id, const ActorMsg& msg) override;
  void SendSocketMsg(int64_t dst_machine_id, const SocketMsg& msg);

 private:
  struct ReadContext {
    CallBackList cbl;
    std::mutex done_cnt_mtx;
    int8_t done_cnt;
  };
  struct ActorReadContext {
    std::mutex read_ctx_list_mtx;
    std::list<ReadContext*> read_ctx_list;
  };
  EpollCommNet();
  int8_t IncreaseDoneCnt(ReadContext*);
  void FinishOneReadContext(ActorReadContext*, ReadContext*);
  void InitSockets();
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

#endif  // ONEFLOW_CORE_COMM_NETWORK_EPOLL_EPOLL_COMM_NETWORK_H_
