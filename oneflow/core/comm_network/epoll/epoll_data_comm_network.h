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
  ~EpollDataCommNet();

  static EpollDataCommNet* Singleton() {
    return static_cast<EpollDataCommNet*>(DataCommNet::Singleton());
  }

  static void Init();

  const void* RegisterMemory(void* mem_ptr, size_t byte_size) override;
  void UnRegisterMemory(const void* token) override;
  void RegisterMemoryDone() override;

  void* Read(int64_t src_machine_id, const void* src_token,
             const void* dst_token) override;
  void AddReadCallBack(void* read_id, std::function<void()> callback) override;
  void AddReadCallBackDone(void* read_id) override;
  void ReadDone(void* read_id);

  void SendActorMsg(int64_t dst_machine_id, const ActorMsg& msg) override;
  void SendSocketMsg(int64_t dst_machine_id, const SocketMsg& msg);

 private:
  EpollDataCommNet();
  void IncreaseDoneCnt(void* read_id);
  void InitSockets();
  SocketHelper* GetSocketHelper(int64_t machine_id);

  // Memory Desc
  std::mutex mem_desc_mtx_;
  std::list<SocketMemDesc*> mem_descs_;
  size_t unregister_mem_descs_cnt_;
  // Read
  struct ReadContext {
    std::mutex cbl_mtx;
    CallBackList cbl;
    std::mutex done_cnt_mtx;
    int8_t done_cnt;
  };
  struct CallBackContext {
    void DecreaseCnt();
    std::function<void()> callback;
    std::mutex cnt_mtx;
    int32_t cnt;
  };
  std::mutex undeleted_read_ctxs_mtx_;
  HashSet<ReadContext*> undeleted_read_ctxs_;
  // Socket
  std::vector<IOEventPoller*> pollers_;
  std::vector<int> machine_id2sockfd_;
  HashMap<int, SocketHelper*> sockfd2helper_;
};

}  // namespace oneflow

#endif  // PLATFORM_POSIX

#endif  // ONEFLOW_CORE_COMM_NETWORK_EPOLL_EPOLL_DATA_COMM_NETWORK_H_
