#include "oneflow/core/comm_network/iocp/io_worker.h"
#include "oneflow/core/job/resource.pb.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/runtime_context.h"
#include "oneflow/core/control/ctrl_client.h"

#ifdef PLATFORM_WINDOWS

namespace oneflow {

namespace {

sockaddr_in GetSockAddr(int64_t machine_id, uint16_t port) {
  const Machine& machine = JobDesc::Singleton()->resource().machine(machine_id);
  const std::string& addr = machine.addr();
  sockaddr_in sa;
  sa.sin_family = AF_INET;
  sa.sin_port = htons(port);
  PCHECK(inet_pton(AF_INET, addr.c_str(), &(sa.sin_addr)) == 1);
  return sa;
}

int64_t GetMachineId(const sockaddr_in& sa) {
  char addr[INET_ADDRSTRLEN];
  memset(addr, '\0', sizeof(addr));
  PCHECK(inet_ntop(AF_INET, (void*)(&(sa.sin_addr)), addr, INET_ADDRSTRLEN));
  for(int64_t i = 0; i < JobDesc::Singleton()->TotalMachineNum(); ++i) {
    if(JobDesc::Singleton()->resource().machine(i).addr() == addr) {
      return i;
    }
  }
  UNEXPECTED_RUN();
}

}  // namespace

IOWorker::IOWorker() {
  num_of_concurrent_threads_ = JobDesc::Singleton()->CommNetIOWorkerNum();
  this_machine_id_ = RuntimeCtx::Singleton()->this_machine_id();
  total_machine_num_ = JobDesc::Singleton()->TotalMachineNum();
  // load winsock
  WSADATA wsd;
  PCHECK(WSAStartup(MAKEWORD(2, 2), &wsd) == 0) << "Unable to load Winsock.\n";
  // create completion port and start N worker thread for it
  // N = num_of_concurrent_threads_
  completion_port_ = CreateIoCompletionPort(INVALID_HANDLE_VALUE, NULL, 0, num_of_concurrent_threads_);
  PCHECK(completion_port_ != NULL) << "CreateIoCompletionPort failed. Error:" << GetLastError() << "\n";
  InitSockets();
}

IOWorker::~IOWorker() {
  for(int64_t i = 0; i < total_machine_num_; ++i) {
    if(i != this_machine_id_) {
      PCHECK(closesocket(machine_id2socket_[i]) == 0);
    }
  }
  WSACleanup();
}

void IOWorker::Start() {
  for(size_t i = 0; i < num_of_concurrent_threads_; ++i) {
    HANDLE worker_thread_handle = CreateThread(NULL, 0, IOWorker::StartThreadProc, this, 0, NULL);
    PCHECK(worker_thread_handle != NULL) << "Create Thread Handle failed. Error:" << GetLastError() << "\n";
    CloseHandle(worker_thread_handle);
  }
}

void IOWorker::Stop() {
  IOData* stop_io_data = new IOData;
  stop_io_data->IO_type = IOType::kStop;
  PostQueuedCompletionStatus(completion_port_, 0, machine_id2socket_[this_machine_id_], reinterpret_cast<LPOVERLAPPED>(stop_io_data));
}

void IOWorker::InitSockets() {
  machine_id2socket_.assign(total_machine_num_, -1);
  // listen
  SOCKET listen_socket = socket(AF_INET, SOCK_STREAM, 0);
  PCHECK(listen_socket != INVALID_SOCKET) << "socket failed with error:" << WSAGetLastError() << "\n";
  machine_id2socket_[this_machine_id_] = listen_socket;
  uint16_t this_listen_port = 1024;
  uint16_t listen_port_max = std::numeric_limits<uint16_t>::max();
  for(; this_listen_port < listen_port_max; ++this_listen_port) {
    sockaddr_in this_sockaddr = GetSockAddr(this_machine_id_, this_listen_port);
    int bind_result =
      bind(listen_socket, reinterpret_cast<sockaddr*>(&this_sockaddr),
           sizeof(this_sockaddr));
    if(bind_result == 0) {
      PCHECK(listen(listen_socket, total_machine_num_) == 0);
      CtrlClient::Singleton()->PushPort(this_listen_port);
      break;
    } else {
      PCHECK(errno == EACCES || errno == EADDRINUSE);
    }
  }
  CHECK_LT(this_listen_port, listen_port_max);
  // connect
  FOR_RANGE(int64_t, peer_machine_id, this_machine_id_ + 1, total_machine_num_) {
    uint16_t peer_port = CtrlClient::Singleton()->PullPort(peer_machine_id);
    sockaddr_in peer_sockaddr = GetSockAddr(peer_machine_id, peer_port);
    SOCKET s = socket(AF_INET, SOCK_STREAM, 0);
    PCHECK(connect(s, reinterpret_cast<sockaddr*>(&peer_sockaddr),
                   sizeof(peer_sockaddr))
           == 0);
    machine_id2socket_[peer_machine_id] = s;
    PCHECK(CreateIoCompletionPort((HANDLE)s, completion_port_, s, 0) != NULL) << "bind to completion port err:" << GetLastError() << "\n";
  }
  // accept
  FOR_RANGE(int64_t, idx, 0, this_machine_id_) {
    sockaddr_in peer_sockaddr;
    socklen_t len = sizeof(peer_sockaddr);
    SOCKET s = accept(listen_socket,
                      reinterpret_cast<sockaddr*>(&peer_sockaddr), &len);
    PCHECK(s != INVALID_SOCKET) << "socket accept error: " << WSAGetLastError() << "\n";
    int64_t peer_machine_id = GetMachineId(peer_sockaddr);
    machine_id2socket_[peer_machine_id] = s;
    PCHECK(CreateIoCompletionPort((HANDLE)s, completion_port_, s, 0) != NULL) << "bind to completion port err:" << GetLastError() << "\n";
  }
  PCHECK(close(listen_socket) == 0);
  // useful log
  FOR_RANGE(int64_t, machine_id, 0, total_machine_num_) {
    LOG(INFO) << "machine " << machine_id << " sockfd "
      << machine_id2socket_[machine_id];
  }
}
   /*
   void IOCPCommNet::SendSocketMsg(int64_t dst_machine_id, const SocketMsg& msg) {
   IOData* io_data_ptr = new IOData;
   memset(&(io_data_ptr->overlapped), 0,sizeof(OVERLAPPED));
   io_data_ptr->socket_msg = msg;
   io_data_ptr->IO_type = IOType::kMsgHead;
   io_data_ptr->data_buff.buf = reinterpret_cast<char*>(&(io_data_ptr->socket_msg));
   io_data_ptr->data_buff.len = sizeof(SocketMsg);
   // WSASend(machine_id2socket_[dst_machine_id], &(io_data_ptr->data_buff), 1, NULL, 0, reinterpret_cast<LPOVERLAPPED>(io_data_ptr), CompletionROUTINE);
   }
   */

   /*
   num_of_concurrent_threads_ = JobDesc::Singleton()->CommNetIOWorkerNum();
   this_machine_id_ = RuntimeCtx::Singleton()->this_machine_id();
   total_machine_num_ = JobDesc::Singleton()->TotalMachineNum();
   // create completion port and start N worker thread for it
   // N = num_of_concurrent_threads_
   completion_port_ = CreateIoCompletionPort(INVALID_HANDLE_VALUE, NULL, 0, num_of_concurrent_threads_);
   CHECK(completion_port_ != NULL) << "CreateIoCompletionPort failed. Error:" << GetLastError() << "\n";
   for(size_t i = 0; i < num_of_concurrent_threads_; ++i) {
   HANDLE worker_thread_handle = CreateThread(NULL, 0, IOworkerThread, completion_port_, 0, NULL);
   CHECK(worker_thread_handle != NULL) << "Create Thread Handle failed. Error:" << GetLastError() << "\n";
   CloseHandle(worker_thread_handle);
   }

   InitSockets();
   */

   /*
   void IOCPCommNet::InitSockets() {
   machine_id2socket_.assign(total_machine_num_, -1);
   // listen
   SOCKET listen_sockfd = socket(AF_INET, SOCK_STREAM, 0);
   uint16_t this_listen_port = 1024;
   uint16_t listen_port_max = std::numeric_limits<uint16_t>::max();
   for(; this_listen_port < listen_port_max; ++this_listen_port) {
   sockaddr_in this_sockaddr = GetSockAddr(this_machine_id_, this_listen_port);
   int bind_result =
   bind(listen_sockfd, reinterpret_cast<sockaddr*>(&this_sockaddr),
   sizeof(this_sockaddr));
   if(bind_result == 0) {
   PCHECK(listen(listen_sockfd, total_machine_num_) == 0);
   CtrlClient::Singleton()->PushPort(this_listen_port);
   break;
   } else {
   PCHECK(errno == EACCES || errno == EADDRINUSE);
   }
   }
   CHECK_LT(this_listen_port, listen_port_max);
   // connect
   FOR_RANGE(int64_t, peer_machine_id, this_machine_id_ + 1, total_machine_num_) {
   uint16_t peer_port = CtrlClient::Singleton()->PullPort(peer_machine_id);
   sockaddr_in peer_sockaddr = GetSockAddr(peer_machine_id, peer_port);
   SOCKET s = socket(AF_INET, SOCK_STREAM, 0);
   PCHECK(connect(s, reinterpret_cast<sockaddr*>(&peer_sockaddr),
   sizeof(peer_sockaddr))
   == 0);
   machine_id2socket_[peer_machine_id] = s;
   }
   // accept
   FOR_RANGE(int64_t, idx, 0, this_machine_id_) {
   sockaddr_in peer_sockaddr;
   socklen_t len = sizeof(peer_sockaddr);
   SOCKET s = accept(listen_sockfd,
   reinterpret_cast<sockaddr*>(&peer_sockaddr), &len);
   PCHECK(s != INVALID_SOCKET) << "socket accept error: " << WSAGetLastError() << "\n";
   int64_t peer_machine_id = GetMachineId(peer_sockaddr);
   machine_id2socket_[peer_machine_id] = s;
   }
   PCHECK(close(listen_sockfd) == 0);
   // bind to completion port
   FOR_RANGE(int64_t, machine_id, 0, total_machine_num_) {
   SOCKET s = machine_id2socket_[machine_id];
   CHECK(CreateIoCompletionPort((HANDLE)s,completion_port_,s,0) != NULL) << "bind to completion port err:" << GetLastError() << "\n";
   LOG(INFO) << "machine " << machine_id << " sockfd "
   << s;
   }
   }
   */


}  // namespace oneflow

#endif  // PLATFORM_WINDOW
