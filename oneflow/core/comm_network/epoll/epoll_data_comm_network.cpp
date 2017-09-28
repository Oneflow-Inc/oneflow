#include "oneflow/core/comm_network/epoll/epoll_data_comm_network.h"
#include "oneflow/core/job/runtime_context.h"

#ifdef PLATFORM_POSIX

namespace oneflow {

namespace {

sockaddr_in GetSockAddr(int64_t machine_id) {
  const Machine& machine = JobDesc::Singleton()->resource().machine(machine_id);
  const std::string& addr = machine.addr();
  uint16_t port = oneflow_cast<uint16_t>(machine.data_port());
  sockaddr_in sa;
  sa.sin_family = AF_INET;
  sa.sin_port = htons(port);
  PCHECK(inet_pton(AF_INET, addr.c_str(), &(sa.sin_addr)) == 1);
  return sa;
}

int64_t GetMachineId(const sockaddr_in& sa) {
  char addr[INET_ADDRSTRLEN];
  memset(addr, '\0', sizeof(addr));
  PCHECK(inet_ntop(AF_INET, &(sa.sin_addr), addr, INET_ADDRSTRLEN));
  for (int64_t i = 0; i < JobDesc::Singleton()->TotalMachineNum(); ++i) {
    if (JobDesc::Singleton()->resource().machine(i).addr() == addr) {
      return i;
    }
  }
  UNEXPECTED_RUN();
}

}  // namespace

EpollDataCommNet::~EpollDataCommNet() {
  for (IOEventPoller* poller : pollers_) { delete poller; }
  for (auto& pair : sockfd2helper_) { delete pair.second; }
}

void EpollDataCommNet::Init() {
  DataCommNet::Singleton()->set_comm_network_ptr(new EpollDataCommNet());
}

const void* EpollDataCommNet::RegisterMemory(void* mem_ptr, size_t byte_size) {
  auto mem_desc = new SocketMemDesc;
  mem_desc->mem_ptr = mem_ptr;
  mem_desc->byte_size = byte_size;
  {
    std::unique_lock<std::mutex> lck(mem_desc_mtx_);
    mem_descs_.push_back(mem_desc);
  }
  return mem_desc;
}

void EpollDataCommNet::UnRegisterMemory(const void* token) {
  std::unique_lock<std::mutex> lck(mem_desc_mtx_);
  CHECK(!mem_descs_.empty());
  unregister_mem_descs_cnt_ += 1;
  if (unregister_mem_descs_cnt_ == mem_descs_.size()) {
    for (SocketMemDesc* mem_desc : mem_descs_) { delete mem_desc; }
    mem_descs_.clear();
    unregister_mem_descs_cnt_ = 0;
  }
}

void EpollDataCommNet::RegisterMemoryDone() {
  // do nothing
}

void* EpollDataCommNet::Read(int64_t src_machine_id, const void* src_token,
                             const void* dst_token) {
  auto callback_list = new CallBackList;
  SocketMsg msg;
  msg.msg_type = SocketMsgType::kRequestWrite;
  msg.request_write_msg.src_token = src_token;
  msg.request_write_msg.dst_machine_id =
      RuntimeCtx::Singleton()->this_machine_id();
  msg.request_write_msg.dst_token = dst_token;
  msg.request_write_msg.read_id = callback_list;
  GetSocketHelper(src_machine_id)->AsyncWrite(msg);
  return callback_list;
}

void EpollDataCommNet::AddReadCallBack(void* read_id,
                                       std::function<void()> callback) {
  auto callback_list = static_cast<CallBackList*>(read_id);
  callback_list->push_back(callback);
}

void EpollDataCommNet::SendActorMsg(int64_t dst_machine_id,
                                    const ActorMsg& actor_msg) {
  SocketMsg msg;
  msg.msg_type = SocketMsgType::kActor;
  msg.actor_msg = actor_msg;
  GetSocketHelper(dst_machine_id)->AsyncWrite(msg);
}

EpollDataCommNet::EpollDataCommNet() {
  mem_descs_.clear();
  unregister_mem_descs_cnt_ = 0;
  pollers_.resize(JobDesc::Singleton()->CommNetIOWorkerNum(), nullptr);
  for (size_t i = 0; i < pollers_.size(); ++i) {
    pollers_[i] = new IOEventPoller;
  }
  InitSockets();
  for (IOEventPoller* poller : pollers_) { poller->Start(); }
}

void EpollDataCommNet::InitSockets() {
  int64_t this_machine_id = RuntimeCtx::Singleton()->this_machine_id();
  int64_t total_machine_num = JobDesc::Singleton()->TotalMachineNum();
  machine_id2sockfd_.assign(total_machine_num, -1);
  sockfd2helper_.clear();
  size_t poller_idx = 0;
  auto NewSocketHelper = [&](int sockfd) {
    IOEventPoller* poller = pollers_[poller_idx];
    poller_idx = (poller_idx + 1) % pollers_.size();
    return new SocketHelper(sockfd, poller);
  };
  // listen
  sockaddr_in this_sockaddr = GetSockAddr(this_machine_id);
  int listen_sockfd = socket(AF_INET, SOCK_STREAM, 0);
  PCHECK(bind(listen_sockfd, reinterpret_cast<sockaddr*>(&this_sockaddr),
              sizeof(this_sockaddr))
         == 0);
  PCHECK(listen(listen_sockfd, total_machine_num) == 0);
  // connect
  FOR_RANGE(int64_t, peer_machine_id, this_machine_id + 1, total_machine_num) {
    sockaddr_in peer_sockaddr = GetSockAddr(peer_machine_id);
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    int rc = -1;
    while (rc == -1) {
      connect(sockfd, reinterpret_cast<sockaddr*>(&peer_sockaddr),
              sizeof(peer_sockaddr));
    }
    PCHECK(rc == 0);
    CHECK(sockfd2helper_.emplace(sockfd, NewSocketHelper(sockfd)).second);
    machine_id2sockfd_[peer_machine_id] = sockfd;
  }
  // accept
  FOR_RANGE(int64_t, idx, 0, this_machine_id) {
    sockaddr_in peer_sockaddr;
    socklen_t len = sizeof(peer_sockaddr);
    int sockfd = accept(listen_sockfd,
                        reinterpret_cast<sockaddr*>(&peer_sockaddr), &len);
    PCHECK(sockfd != -1);
    CHECK(sockfd2helper_.emplace(sockfd, NewSocketHelper(sockfd)).second);
    int64_t peer_machine_id = GetMachineId(peer_sockaddr);
    machine_id2sockfd_[peer_machine_id] = sockfd;
  }
  PCHECK(close(listen_sockfd) == 0);
}

SocketHelper* EpollDataCommNet::GetSocketHelper(int64_t machine_id) {
  int sockfd = machine_id2sockfd_.at(machine_id);
  return sockfd2helper_.at(sockfd);
}

}  // namespace oneflow

#endif  // PLATFORM_POSIX
