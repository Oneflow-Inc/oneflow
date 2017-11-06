#ifndef ONEFLOW_CORE_COMM_NETWORK_IOCP_IOCP_COMM_NETWORK_H_
#define ONEFLOW_CORE_COMM_NETWORK_IOCP_IOCP_COMM_NETWORK_H_

#include "oneflow/core/comm_network/comm_network.h"
#include "oneflow/core/comm_network/iocp/io_type.h"
#include "oneflow/core/comm_network/iocp/io_worker.h"

#ifdef PLATFORM_WINDOWS

namespace oneflow {

class IOCPCommNet final : public CommNet {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IOCPCommNet);
  ~IOCPCommNet();

  static IOCPCommNet* Singleton() {
    return static_cast<IOCPCommNet*>(CommNet::Singleton());
  }

  static void Init();

  const void* RegisterMemory(void* mem_ptr, size_t byte_size) override;
  void UnRegisterMemory(const void* token) override;
  void RegisterMemoryDone() override;

  void* NewActorReadId() override;
  void DeleteActorReadId(void* actor_read_id) override;
  void* Read(void* actor_read_id, int64_t write_machine_id,
             const void* write_token, const void* read_token) override;
  void AddReadCallBack(void* actor_read_id, void* read_id,
                       std::function<void()> callback) override;
  void AddReadCallBackDone(void* actor_read_id, void* read_id) override;
  void ReadDone(void* read_done_id);

  void SendActorMsg(int64_t dst_machine_id, const ActorMsg& msg) override;

 private:
  IOCPCommNet();
  int8_t IncreaseDoneCnt(ReadContext*);
  void FinishOneReadContext(ActorReadContext*, ReadContext*);

  // Memory Desc
  std::mutex mem_desc_mtx_;
  std::list<SocketMemDesc*> mem_descs_;
  size_t unregister_mem_descs_cnt_;

  // io_worker
  IOWorker* io_worker_ptr_;
};

}  // namespace oneflow

#endif  // PLATFORM_WINDOWS

#endif  //  ONEFLOW_CORE_COMM_NETWORK_IOCP_IOCP_COMM_NETWORK_H_
