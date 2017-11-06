#include "oneflow/core/comm_network/iocp/io_worker.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/resource.pb.h"
#include "oneflow/core/job/runtime_context.h"

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
  for (int64_t i = 0; i < JobDesc::Singleton()->TotalMachineNum(); ++i) {
    if (JobDesc::Singleton()->resource().machine(i).addr() == addr) {
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
  completion_port_ = CreateIoCompletionPort(INVALID_HANDLE_VALUE, NULL, 0,
                                            num_of_concurrent_threads_);
  PCHECK(completion_port_ != NULL)
      << "CreateIoCompletionPort failed. Error:" << GetLastError() << "\n";

  InitSockets();

  // post WSARecv to socket for recv msg from remote
  machine_id2io_data_recv_.clear();
  for (int64_t i = 0; i < total_machine_num_; ++i) {
    machine_id2io_data_recv_.push_back(new IOData);
  }
  for (int64_t i = 0; i < total_machine_num_; ++i) {
    if (i != this_machine_id_) { PostNewWSARecv2Socket(i); }
  }

  // init send queue vector
  machine_id2io_data_send_que_.clear();
  for (int64_t i = 0; i < total_machine_num_; ++i) {
    std::queue<IOData*> q;
    machine_id2io_data_send_que_.push_back(q);
  }
}

IOWorker::~IOWorker() {
  for (int64_t i = 0; i < total_machine_num_; ++i) {
    if (i != this_machine_id_) {
      PCHECK(closesocket(machine_id2socket_[i]) == 0);
    }
  }
  WSACleanup();
}

void IOWorker::PostSendMsgRequest(int64_t dst_machine_id,
                                  SocketMsg socket_msg) {
  SOCKET s = machine_id2socket_[dst_machine_id];
  IOData* io_data_ptr = new IOData;
  memset(&(io_data_ptr->overlapped), 0, sizeof(OVERLAPPED));
  io_data_ptr->socket_msg = socket_msg;
  io_data_ptr->IO_type = IOType::kSendMsgHead;
  io_data_ptr->data_buff.buf =
      reinterpret_cast<char*>(&(io_data_ptr->socket_msg));
  io_data_ptr->data_buff.len = sizeof(SocketMsg);
  io_data_ptr->target_machine_id = dst_machine_id;
  io_data_ptr->target_socket_fd = s;
  io_data_ptr->flags = 0;
  PostQueuedCompletionStatus(completion_port_, 0, s,
                             reinterpret_cast<LPOVERLAPPED>(io_data_ptr));
  // WSASend(machine_id2socket_[dst_machine_id], &(io_data_ptr->data_buff), 1,
  // NULL, 0, reinterpret_cast<LPOVERLAPPED>(io_data_ptr), CompletionROUTINE);
}

void IOWorker::Start() {
  for (size_t i = 0; i < num_of_concurrent_threads_; ++i) {
    HANDLE worker_thread_handle =
        CreateThread(NULL, 0, IOWorker::StartThreadProc, this, 0, NULL);
    PCHECK(worker_thread_handle != NULL)
        << "Create Thread Handle failed. Error:" << GetLastError() << "\n";
    CloseHandle(worker_thread_handle);
  }
}

void IOWorker::Stop() {
  IOData* stop_io_data = new IOData;
  stop_io_data->IO_type = IOType::kStop;
  PostQueuedCompletionStatus(completion_port_, 0,
                             machine_id2socket_[this_machine_id_],
                             reinterpret_cast<LPOVERLAPPED>(stop_io_data));
}

void IOWorker::InitSockets() {
  machine_id2socket_.clear();
  machine_id2socket_.assign(total_machine_num_, -1);
  // listen
  SOCKET listen_socket = socket(AF_INET, SOCK_STREAM, 0);
  PCHECK(listen_socket != INVALID_SOCKET)
      << "socket failed with error:" << WSAGetLastError() << "\n";
  machine_id2socket_[this_machine_id_] = listen_socket;
  uint16_t this_listen_port = 1024;
  uint16_t listen_port_max = std::numeric_limits<uint16_t>::max();
  for (; this_listen_port < listen_port_max; ++this_listen_port) {
    sockaddr_in this_sockaddr = GetSockAddr(this_machine_id_, this_listen_port);
    int bind_result =
        bind(listen_socket, reinterpret_cast<sockaddr*>(&this_sockaddr),
             sizeof(this_sockaddr));
    if (bind_result == 0) {
      PCHECK(listen(listen_socket, total_machine_num_) == 0);
      CtrlClient::Singleton()->PushPort(this_listen_port);
      break;
    } else {
      PCHECK(errno == EACCES || errno == EADDRINUSE);
    }
  }
  CHECK_LT(this_listen_port, listen_port_max);
  // connect
  FOR_RANGE(int64_t, peer_machine_id, this_machine_id_ + 1,
            total_machine_num_) {
    uint16_t peer_port = CtrlClient::Singleton()->PullPort(peer_machine_id);
    sockaddr_in peer_sockaddr = GetSockAddr(peer_machine_id, peer_port);
    SOCKET s = socket(AF_INET, SOCK_STREAM, 0);
    PCHECK(connect(s, reinterpret_cast<sockaddr*>(&peer_sockaddr),
                   sizeof(peer_sockaddr))
           == 0);
    machine_id2socket_[peer_machine_id] = s;
    PCHECK(CreateIoCompletionPort((HANDLE)s, completion_port_, s, 0) != NULL)
        << "bind to completion port err:" << GetLastError() << "\n";
  }
  // accept
  FOR_RANGE(int64_t, idx, 0, this_machine_id_) {
    sockaddr_in peer_sockaddr;
    socklen_t len = sizeof(peer_sockaddr);
    SOCKET s = accept(listen_socket,
                      reinterpret_cast<sockaddr*>(&peer_sockaddr), &len);
    PCHECK(s != INVALID_SOCKET)
        << "socket accept error: " << WSAGetLastError() << "\n";
    int64_t peer_machine_id = GetMachineId(peer_sockaddr);
    machine_id2socket_[peer_machine_id] = s;
    PCHECK(CreateIoCompletionPort((HANDLE)s, completion_port_, s, 0) != NULL)
        << "bind to completion port err:" << GetLastError() << "\n";
  }
  PCHECK(close(listen_socket) == 0);
  // useful log
  FOR_RANGE(int64_t, machine_id, 0, total_machine_num_) {
    LOG(INFO) << "machine " << machine_id << " sockfd "
              << machine_id2socket_[machine_id];
  }
}

void IOWorker::PostNewWSARecv2Socket(int64_t dst_machine_id) {
  SOCKET s = machine_id2socket_[dst_machine_id];
  IOData* io_data_ptr = machine_id2io_data_recv_[dst_machine_id];
  memset(&(io_data_ptr->overlapped), 0, sizeof(OVERLAPPED));
  io_data_ptr->IO_type = IOType::kRecvMsgHead;
  io_data_ptr->data_buff.buf =
      reinterpret_cast<char*>(&(io_data_ptr->socket_msg));
  io_data_ptr->data_buff.len = sizeof(SocketMsg);
  io_data_ptr->target_machine_id = dst_machine_id;
  io_data_ptr->target_socket_fd = s;
  io_data_ptr->flags = 0;
  WSARecv(s, &(io_data_ptr->data_buff), 1, NULL, &(io_data_ptr->flags),
          reinterpret_cast<LPOVERLAPPED>(io_data_ptr), NULL);
}

DWORD IOWorker::ThreadProc() {
  DWORD bytes_transferred;
  SOCKET completion_key;
  IOData* io_data_ptr;
  while (true) {
    CHECK(GetQueuedCompletionStatus(
              completion_port_, &bytes_transferred, &completion_key,
              reinterpret_cast<LPOVERLAPPED*>(&io_data_ptr), INFINITE)
          == true)
        << "GetQueuedCompletionStatus Error: " << GetLastError() << "\n";
    switch (io_data_ptr->IO_type) {
      case IOType::kStop: {
        delete io_data_ptr;
        return 0;
      }
      case IOType::kRecvMsgHead: {
        OnRecvMsgHead(bytes_transferred, io_data_ptr);
        break;
      }
      case IOType::kRecvMsgBody: {
        OnRecvMsgBody(bytes_transferred, io_data_ptr);
        break;
      }
      case IOType::kSendMsgHead: {
        OnSendMsgHead(bytes_transferred, io_data_ptr);
        break;
      }
      case IOType::kSendMsgBody: {
        OnSendMsgBody(bytes_transferred, io_data_ptr);
        break;
      }
      default: UNEXPECTED_RUN()
    }
  }
  return 0;
}

void IOWorker::OnRecvMsgHead(DWORD bytes_transferred, IOData* io_data_ptr) {
  // TO DO
}

void IOWorker::OnRecvMsgBody(DWORD bytes_transferred, IOData* io_data_ptr) {
  // TO DO
}

void IOWorker::OnSendMsgHead(DWORD bytes_transferred, IOData* io_data_ptr) {
  // TO DO
}

void IOWorker::OnSendMsgBody(DWORD bytes_transferred, IOData* io_data_ptr) {
  // TO DO
}

}  // namespace oneflow

#endif  // PLATFORM_WINDOW
