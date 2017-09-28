#include "oneflow/core/comm_network/epoll/io_event_poller.h"

#ifdef PLATFORM_POSIX

namespace oneflow {

const int IOEventPoller::max_event_num_ = 32;

IOEventPoller::IOEventPoller() {
  epfd_ = epoll_create1(0);
  ep_events_ = new epoll_event[max_event_num_];
  unclosed_fd_cnt_ = 0;
  io_handlers_.clear();
  fds_.clear();
}

IOEventPoller::~IOEventPoller() {
  for (int fd : fds_) { PCHECK(close(fd) == 0); }
  thread_.join();
  for (IOHandler* handler : io_handlers_) { delete handler; }
  CHECK_EQ(unclosed_fd_cnt_, 0);
  delete[] ep_events_;
  PCHECK(close(epfd_) == 0);
}

void IOEventPoller::AddFd(int fd, std::function<void()> read_handler,
                          std::function<void()> write_handler) {
  unclosed_fd_cnt_ += 1;
  fds_.push_back(fd);
  // Set Fd NONBLOCK
  int opt = fcntl(fd, F_GETFL);
  PCHECK(opt != -1);
  PCHECK(fcntl(fd, F_SETFL, opt | O_NONBLOCK) == 0);
  // New IOHandler on Heap
  IOHandler* io_handler = new IOHandler;
  io_handler->read_handler = read_handler;
  io_handler->write_handler = write_handler;
  io_handlers_.push_front(io_handler);
  // Add Fd to Epoll
  epoll_event ep_event;
  ep_event.events = EPOLLIN | EPOLLOUT | EPOLLET;
  ep_event.data.ptr = io_handler;
  PCHECK(epoll_ctl(epfd_, EPOLL_CTL_ADD, fd, &ep_event) == 0);
}

void IOEventPoller::Start() {
  thread_ = std::thread(&IOEventPoller::EpollLoop, this);
}

void IOEventPoller::EpollLoop() {
  while (unclosed_fd_cnt_ > 0) {
    int event_num = epoll_wait(epfd_, ep_events_, max_event_num_, -1);
    PCHECK(event_num >= 0);
    const epoll_event* cur_event = ep_events_;
    for (int event_idx = 0; event_idx < event_num; ++event_idx, ++cur_event) {
      PCHECK(!(cur_event->events & EPOLLERR));
      auto io_handler = static_cast<IOHandler*>(cur_event->data.ptr);
      if (cur_event->events & EPOLLIN) {
        if (cur_event->events & EPOLLRDHUP) { unclosed_fd_cnt_ -= 1; }
        io_handler->read_handler();
      }
      if (cur_event->events & EPOLLOUT) { io_handler->write_handler(); }
    }
  }
}

}  // namespace oneflow

#endif  // PLATFORM_POSIX
