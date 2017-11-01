#ifndef  ONEFLOW_CORE_COMM_NETWORK_IOCP_IO_TYPE_H_
#define ONEFLOW_CORE_COMM_NETWORK_IOCP_IO_TYPE_H_

#ifdef PLATFORM_WINDOWS

#include <mutex>
#include <WinSock2.h>
#include <Windows.h>
#include <WinBase.h>
#include <WS2tcpip.h>

#pragma comment(lib, "Ws2_32.lib")
#pragma comment(lib, "Kernel32.lib")

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

struct ReadContext {
  CallBackList cbl;
  std::mutex done_cnt_mtx;
  int8_t done_cnt;
};
struct ActorReadContext {
  std::mutex read_ctx_list_mtx;
  std::list<ReadContext*> read_ctx_list;
};

using CallBackList = std::list<std::function<void()>>;

}  // namespace oneflow

#endif  // PLATFORM_WINDOWS

#endif  //  ONEFLOW_CORE_COMM_NETWORK_IOCP_IO_TYPE_H
