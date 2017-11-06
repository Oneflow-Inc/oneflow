#ifndef ONEFLOW_CORE_COMM_NETWORK_IOCP_IO_TYPE_H_
#define ONEFLOW_CORE_COMM_NETWORK_IOCP_IO_TYPE_H_

#ifdef PLATFORM_WINDOWS

#include <WS2tcpip.h>
#include <WinBase.h>
#include <WinSock2.h>
#include <Windows.h>
#include <mutex>
#include "oneflow/core/actor/actor_message.h"

#pragma comment(lib, "Ws2_32.lib")
#pragma comment(lib, "Kernel32.lib")

namespace oneflow {

struct SocketMemDesc {
  void* mem_ptr;
  size_t byte_size;
};

enum IOType {
  kSendMsgHead,
  kSendMsgBody,
  kRecvMsgHead,
  kRecvMsgBody,
  kStop,
};

enum SocketMsgType {
  kRequestWrite,
  kRequsetRead,
  kActor,
};

struct SocketToken {
  const void* write_machine_mem_desc_;
  const void* read_machine_mem_desc_;
  void* read_done_id;
};

struct SocketMsg {
  SocketMsgType msg_type;
  union {
    SocketToken socket_token;
    ActorMsg actor_msg;
  };
};

struct IOData {
  OVERLAPPED overlapped;
  IOType IO_type;
  WSABUF data_buff;
  DWORD flags;
  SocketMsg socket_msg;
  SOCKET target_socket_fd;
  int64_t target_machine_id;
};

using CallBackList = std::list<std::function<void()>>;

struct ReadContext {
  CallBackList cbl;
  std::mutex done_cnt_mtx;
  int8_t done_cnt;
};
struct ActorReadContext {
  std::mutex read_ctx_list_mtx;
  std::list<ReadContext*> read_ctx_list;
};

using ReadDoneContext = std::tuple<ActorReadContext*, ReadContext*>;

}  // namespace oneflow

#endif  // PLATFORM_WINDOWS

#endif  //  ONEFLOW_CORE_COMM_NETWORK_IOCP_IO_TYPE_H
