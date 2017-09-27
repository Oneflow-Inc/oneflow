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

EpollDataCommNet::~EpollDataCommNet() {
  for (int sockfd : machine_id2sockfd_) { PCHECK(close(sockfd) == 0); }
  epoll_thread_.join();
  for (SocketIOWorker* worker : io_workers_) { delete worker; }
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
  GetSocketWriteHelper(src_machine_id)->Write(msg);
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
  GetSocketWriteHelper(dst_machine_id)->Write(msg);
}

EpollDataCommNet::EpollDataCommNet() {
  mem_descs_.clear();
  unregister_mem_descs_cnt_ = 0;
  io_workers_.resize(JobDesc::Singleton()->CommNetIOWorkerNum(), nullptr);
  for (size_t i = 0; i < io_workers_.size(); ++i) {
    io_workers_[i] = new SocketIOWorker;
  }
  InitSockets();
  epoll_thread_ = std::thread(&EpollDataCommNet::EpollLoop, this);
}

void EpollDataCommNet::InitSockets() {
  int64_t this_machine_id = RuntimeCtx::Singleton()->this_machine_id();
  int64_t total_machine_num = JobDesc::Singleton()->TotalMachineNum();
  machine_id2sockfd_.assign(total_machine_num, -1);
  sockfd2io_helper_.clear();
  size_t worker_idx = 0;
  auto NewSocketIOHelper = [&](int sockfd) {
    SocketIOWorker* reader = io_workers_[worker_idx];
    worker_idx = (worker_idx + 1) % io_workers_.size();
    SocketIOWorker* writer = io_workers_[worker_idx];
    worker_idx = (worker_idx + 1) % io_workers_.size();
    return of_make_unique<SocketHelper>(sockfd, reader, writer);
  };
  // listen
  sockaddr_in this_sockaddr = GetSockAddr(this_machine_id);
  int listen_sockfd = socket(AF_INET, SOCK_STREAM, 0);
  PCHECK(bind(listen_sockfd, reinterpret_cast<sockaddr*>(&this_sockaddr),
              sizeof(this_sockaddr))
         == 0);
  PCHECK(listen(listen_sockfd, total_machine_num) == 0);
  machine_id2sockfd_[this_machine_id] = listen_sockfd;
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
    CHECK(sockfd2io_helper_.emplace(sockfd, NewSocketIOHelper(sockfd)).second);
    machine_id2sockfd_[peer_machine_id] = sockfd;
  }
  // accept
  FOR_RANGE(int64_t, peer_machine_id, 0, this_machine_id) {
    sockaddr_in peer_sockaddr = GetSockAddr(peer_machine_id);
    socklen_t len = sizeof(peer_sockaddr);
    int sockfd = accept(listen_sockfd,
                        reinterpret_cast<sockaddr*>(&peer_sockaddr), &len);
    PCHECK(sockfd != -1);
    CHECK(sockfd2io_helper_.emplace(sockfd, NewSocketIOHelper(sockfd)).second);
    machine_id2sockfd_[peer_machine_id] = sockfd;
  }
}

void EpollDataCommNet::EpollLoop() {
  int epfd = epoll_create1(0);
  PCHECK(epfd != -1);
  for (auto& pair : sockfd2io_helper_) {
    epoll_event ep_event;
    ep_event.events = EPOLLIN | EPOLLOUT | EPOLLET;
    ep_event.data.ptr = pair.second.get();
    PCHECK(epoll_ctl(epfd, EPOLL_CTL_ADD, pair.first, &ep_event) == 0);
  }
  const int maxevents = 32;
  std::vector<epoll_event> ep_events(maxevents);
  int64_t rdhup_cnt = 0;
  while (rdhup_cnt < JobDesc::Singleton()->TotalMachineNum() - 1) {
    int event_num = epoll_wait(epfd, &ep_events[0], maxevents, -1);
    PCHECK(event_num >= 0);
    FOR_RANGE(int, event_idx, 0, event_num) {
      const epoll_event& cur_event = ep_events[event_idx];
      auto io_helper = static_cast<SocketHelper*>(cur_event.data.ptr);
      PCHECK(!(cur_event.events & EPOLLERR));
      if (cur_event.events & EPOLLIN) {
        if (cur_event.events & EPOLLRDHUP) { rdhup_cnt += 1; }
        io_helper->mut_read_helper()->NotifyWorker();
      }
      if (cur_event.events & EPOLLOUT) {
        io_helper->mut_write_helper()->NotifyWorker();
      }
    }
  }
}

SocketHelper* EpollDataCommNet::GetSocketHelper(int64_t machine_id) {
  int sockfd = machine_id2sockfd_.at(machine_id);
  return sockfd2io_helper_.at(sockfd).get();
}

SocketReadHelper* EpollDataCommNet::GetSocketReadHelper(int64_t machine_id) {
  return GetSocketHelper(machine_id)->mut_read_helper();
}

SocketWriteHelper* EpollDataCommNet::GetSocketWriteHelper(int64_t machine_id) {
  return GetSocketHelper(machine_id)->mut_write_helper();
}

}  // namespace oneflow

#endif  // PLATFORM_POSIX
