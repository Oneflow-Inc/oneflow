#include "oneflow/core/comm_network/iocp/iocp_comm_network.h"
#include "oneflow/core/control/ctrl_client.h"
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
  for(int64_t i = 0; i < JobDesc::Singleton()->TotalMachineNum(); ++i) {
    if(JobDesc::Singleton()->resource().machine(i).addr() == addr) {
      return i;
    }
  }
  UNEXPECTED_RUN();
}

DWORD WINAPI IOworkerThread(LPVOID completion_port_id) {}

void CALLBACK CompletionROUTINE(
  IN DWORD dwError,
  IN DWORD cbTransferred,
  IN LPWSAOVERLAPPED lpOverlapped,
  IN DWORD dwFlags
);

}  // namespace

IOCPCommNet::~IOCPCommNet() {
  Stop();
  OF_BARRIER();
}

void IOCPCommNet::Init() {
  CommNet::Singleton()->set_comm_network_ptr(new IOCPCommNet());
}

const void* IOCPCommNet::RegisterMemory(void* mem_ptr, size_t byte_size) {
  auto mem_desc = new SocketMemDesc;
  mem_desc->mem_ptr = mem_ptr;
  mem_desc->byte_size = byte_size;
  {
    std::unique_lock<std::mutex> lck(mem_desc_mtx_);
    mem_descs_.push_back(mem_desc);
  }
  return mem_desc;
}

void IOCPCommNet::UnRegisterMemory(const void* token) {
  std::unique_lock<std::mutex> lck(mem_desc_mtx_);
  CHECK(!mem_descs_.empty());
  unregister_mem_descs_cnt_ += 1;
  if(unregister_mem_descs_cnt_ == mem_descs_.size()) {
    for(SocketMemDesc* mem_desc : mem_descs_) { delete mem_desc; }
    mem_descs_.clear();
    unregister_mem_descs_cnt_ = 0;
  }
}

void IOCPCommNet::RegisterMemoryDone() {
  // do nothing
}

void* IOCPCommNet::NewActorReadId() { return new ActorReadContext; }

void IOCPCommNet::DeleteActorReadId(void* actor_read_id) {
  auto actor_read_ctx = static_cast<ActorReadContext*>(actor_read_id);
  CHECK(actor_read_ctx->read_ctx_list.empty());
  delete actor_read_ctx;
}

void* IOCPCommNet::Read(void* actor_read_id, int64_t write_machine_id,
                         const void* write_token, const void* read_token) {
  // ReadContext
  auto actor_read_ctx = static_cast<ActorReadContext*>(actor_read_id);
  ReadContext* read_ctx = new ReadContext;
  read_ctx->done_cnt = 0;
  {
    std::unique_lock<std::mutex> lck(actor_read_ctx->read_ctx_list_mtx);
    actor_read_ctx->read_ctx_list.push_back(read_ctx);
  }
  // request write msg
  SocketMsg msg;
  msg.msg_type = SocketMsgType::kRequestWrite;
  msg.request_write_msg.write_token = write_token;
  msg.request_write_msg.read_machine_id = this_machine_id_;
  msg.request_write_msg.read_token = read_token;
  msg.request_write_msg.read_done_id =
    new std::tuple<ActorReadContext*, ReadContext*>(actor_read_ctx, read_ctx);
  SendSocketMsg(write_machine_id, msg);
  return read_ctx;
}

void IOCPCommNet::AddReadCallBack(void* actor_read_id, void* read_id,
                                   std::function<void()> callback) {
  auto actor_read_ctx = static_cast<ActorReadContext*>(actor_read_id);
  ReadContext* read_ctx = static_cast<ReadContext*>(read_id);
  if(read_ctx) {
    read_ctx->cbl.push_back(callback);
    return;
  }
  do {
    std::unique_lock<std::mutex> lck(actor_read_ctx->read_ctx_list_mtx);
    if(actor_read_ctx->read_ctx_list.empty()) {
      break;
    } else {
      actor_read_ctx->read_ctx_list.back()->cbl.push_back(callback);
      return;
    }
  } while(0);
  callback();
}

void IOCPCommNet::AddReadCallBackDone(void* actor_read_id, void* read_id) {
  auto actor_read_ctx = static_cast<ActorReadContext*>(actor_read_id);
  ReadContext* read_ctx = static_cast<ReadContext*>(read_id);
  if(IncreaseDoneCnt(read_ctx) == 2) {
    FinishOneReadContext(actor_read_ctx, read_ctx);
    delete read_ctx;
  }
}

void IOCPCommNet::ReadDone(void* read_done_id) {
  auto parsed_read_done_id =
    static_cast<std::tuple<ActorReadContext*, ReadContext*>*>(read_done_id);
  auto actor_read_ctx = std::get<0>(*parsed_read_done_id);
  auto read_ctx = std::get<1>(*parsed_read_done_id);
  delete parsed_read_done_id;
  if(IncreaseDoneCnt(read_ctx) == 2) {
    {
      std::unique_lock<std::mutex> lck(actor_read_ctx->read_ctx_list_mtx);
      FinishOneReadContext(actor_read_ctx, read_ctx);
    }
    delete read_ctx;
  }
}

int8_t IOCPCommNet::IncreaseDoneCnt(ReadContext* read_ctx) {
  std::unique_lock<std::mutex> lck(read_ctx->done_cnt_mtx);
  read_ctx->done_cnt += 1;
  return read_ctx->done_cnt;
}

void IOCPCommNet::FinishOneReadContext(ActorReadContext* actor_read_ctx,
                                        ReadContext* read_ctx) {
  CHECK_EQ(actor_read_ctx->read_ctx_list.front(), read_ctx);
  actor_read_ctx->read_ctx_list.pop_front();
  for(std::function<void()>& callback : read_ctx->cbl) { callback(); }
}

void IOCPCommNet::SendActorMsg(int64_t dst_machine_id,
                                const ActorMsg& actor_msg) {
  SocketMsg msg;
  msg.msg_type = SocketMsgType::kActor;
  msg.actor_msg = actor_msg;
  SendSocketMsg(dst_machine_id, msg);
}

void IOCPCommNet::SendSocketMsg(int64_t dst_machine_id, const SocketMsg& msg) {
  IOData* io_data_ptr = new IOData;
  memset(&(io_data_ptr->overlapped), 0,sizeof(OVERLAPPED));
  io_data_ptr->socket_msg = msg;
  io_data_ptr->IO_type = IOType::kMsgHead;
  io_data_ptr->data_buff.buf = reinterpret_cast<char*>(&(io_data_ptr->socket_msg));
  io_data_ptr->data_buff.len = sizeof(SocketMsg);
  WSASend(machine_id2socket_[dst_machine_id], &(io_data_ptr->data_buff), 1, NULL, 0, reinterpret_cast<LPOVERLAPPED>(io_data_ptr), CompletionROUTINE);
}

IOCPCommNet::IOCPCommNet() {
  mem_descs_.clear();
  unregister_mem_descs_cnt_ = 0;
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
}

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

}  // namespace oneflow

#endif // PLATFORM_WINDOWS
