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

}  // namespace

void EpollDataCommNet::Init(const Plan& plan) {
  DataCommNet::Singleton()->set_comm_network_ptr(new EpollDataCommNet(plan));
}

const void* EpollDataCommNet::RegisterMemory(void* mem_ptr, size_t byte_size) {
  auto mem_desc = new MemDesc;
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
    for (MemDesc* mem_desc : mem_descs_) { delete mem_desc; }
    mem_descs_.clear();
    unregister_mem_descs_cnt_ = 0;
  }
}

void EpollDataCommNet::RegisterMemoryDone() {
  // do nothing
}

void* EpollDataCommNet::CreateStream() { TODO(); }

void EpollDataCommNet::Read(void* stream_id, const void* src_token,
                            const void* dst_token) {
  TODO();
}

void EpollDataCommNet::AddCallBack(void* stream_id, std::function<void()>) {
  TODO();
}

void EpollDataCommNet::SendActorMsg(int64_t dst_machine_id,
                                    const ActorMsg& msg) {
  TODO();
}

EpollDataCommNet::EpollDataCommNet(const Plan& plan) {
  mem_descs_.clear();
  unregister_mem_descs_cnt_ = 0;
  int64_t this_machine_id = RuntimeCtx::Singleton()->this_machine_id();
  int64_t total_machine_num = JobDesc::Singleton()->TotalMachineNum();
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
    CHECK(socket2io_helper_
              .emplace(sockfd, of_make_unique<SocketIOHelper>(sockfd))
              .second);
  }
  // accept
  FOR_RANGE(int64_t, peer_machine_id, 0, this_machine_id) {
    sockaddr_in peer_sockaddr = GetSockAddr(peer_machine_id);
    socklen_t len = sizeof(peer_sockaddr);
    int sockfd = accept(listen_sockfd,
                        reinterpret_cast<sockaddr*>(&peer_sockaddr), &len);
    PCHECK(sockfd != -1);
    CHECK(socket2io_helper_
              .emplace(sockfd, of_make_unique<SocketIOHelper>(sockfd))
              .second);
  }
}

}  // namespace oneflow

#endif  // PLATFORM_POSIX
