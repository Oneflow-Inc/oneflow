#ifndef  ONEFLOW_CORE_COMM_NETWORK_IOCP_IOCP_COMM_NETWORK_H_
#define ONEFLOW_CORE_COMM_NETWORK_IOCP_IOCP_COMM_NETWORK_H_

#include "oneflow/core/comm_network/comm_network.h"

#ifdef PLATFORM_WINDOWS

#include <mutex>
#include <WinSock2.h>
#include <Windows.h>
#include <WinBase.h>
#include <WS2tcpip.h>

#pragma comment(lib, "Ws2_32.lib")
#pragma comment(lib, "Kernel32.lib")

#define MAX_MSG_SIZE 1024

namespace oneflow {

struct SocketMemDesc {
  void* mem_ptr;
  size_t byte_size;
};

enum IOType {
  kMsgHead,
  kMsgBody,
};

enum SocketMsgType {
  kRequestWrite,
  kRequsetRead,
  kActor,
};

struct RequestWriteMsg {
  const void* write_token;
  int64_t read_machine_id;
  const void* read_token;
  void* read_done_id;
};

struct RequestReadMsg {
  const void* read_token;
  const void* write_token;
  void* read_done_id;
};

struct SocketMsg {
  SocketMsgType msg_type;
  union {
    RequestWriteMsg request_write_msg;
    RequestReadMsg request_read_msg;
    ActorMsg actor_msg;
  };
};

struct IOData {
  OVERLAPPED overlapped;
  IOType IO_type;
  WSABUF data_buff;
  SocketMsg socket_msg;
};

using CallBackList = std::list<std::function<void()>>;

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
  void* Read(void* actor_read_id, int64_t write_machine_id, const void* write_token,
             const void* read_token) override;
  void AddReadCallBack(void* actor_read_id, void* read_id,
                       std::function<void()> callback) override;
  void AddReadCallBackDone(void* actor_read_id, void* read_id) override;
  void ReadDone(void* read_done_id);

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
  IOCPCommNet();
  int8_t IncreaseDoneCnt(ReadContext*);
  void FinishOneReadContext(ActorReadContext*, ReadContext*);
  void InitSockets();
  void Stop();

  // Memory Desc
  std::mutex mem_desc_mtx_;
  std::list<SocketMemDesc*> mem_descs_;
  size_t unregister_mem_descs_cnt_;
  // Socket
  std::vector<SOCKET> machine_id2socket_;

  // completion port
  HANDLE completion_port_;
  int32_t num_of_concurrent_threads_;

  // machine_id
  int64_t this_machine_id_;
  int64_t total_machine_num_;
};

}  // namespace oneflow

#endif  // PLATFORM_WINDOWS

#endif  //  ONEFLOW_CORE_COMM_NETWORK_IOCP_IOCP_COMM_NETWORK_H_
